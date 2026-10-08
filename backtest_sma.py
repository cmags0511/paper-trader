"""
Simple moving-average crossover backtest with an honest train/test split.

Setup (run on your own computer, where price data is reachable):
    pip install yfinance pandas numpy
    python backtest_sma.py
    python backtest_sma.py --tickers NVDA MRVL JPM --years 8 --cost 0.001

What it does:
  1. Downloads daily prices for each ticker.
  2. Tries a grid of fast/slow SMA pairs on the FIRST 70% of history (train).
  3. Applies the single best pair, unchanged, to the LAST 30% (test).
  4. Compares against buy-and-hold on that same test window.

Rules: long when fast SMA > slow SMA, otherwise in cash. Signals are shifted one
day so you never trade on information you wouldn't have had. Costs are charged
on every position change.

This is research code, not financial advice. A strategy that looks good on
training data and fails on test data (common) is not worth trading.
"""
import argparse

import numpy as np
import pandas as pd

DEFAULT_TICKERS = ["NVDA", "MRVL", "JPM", "GS", "DAL", "LEN"]
FAST_GRID = [5, 10, 20, 30]
SLOW_GRID = [50, 100, 150, 200]


def load_prices(ticker, years):
    import yfinance as yf

    df = yf.download(ticker, period=f"{years}y", auto_adjust=True, progress=False)
    if df.empty:
        raise ValueError(f"no data returned for {ticker}")
    close = df["Close"]
    if isinstance(close, pd.DataFrame):  # newer yfinance returns a 1-column frame
        close = close.iloc[:, 0]
    return close.dropna().rename(ticker)


def strategy_returns(close, fast, slow, cost):
    """Daily strategy returns for a long/cash SMA crossover."""
    sma_fast = close.rolling(fast).mean()
    sma_slow = close.rolling(slow).mean()
    position = (sma_fast > sma_slow).astype(float).shift(1).fillna(0.0)
    daily = close.pct_change().fillna(0.0)
    trades = position.diff().abs().fillna(0.0)
    return position * daily - trades * cost, int(trades.sum())


def metrics(returns, trades=0):
    returns = returns.dropna()
    if len(returns) < 2:
        return dict(total=np.nan, cagr=np.nan, sharpe=np.nan, max_dd=np.nan, trades=trades)
    equity = (1 + returns).cumprod()
    years = len(returns) / 252
    total = equity.iloc[-1] - 1
    cagr = equity.iloc[-1] ** (1 / years) - 1 if years > 0 else np.nan
    sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else np.nan
    max_dd = (equity / equity.cummax() - 1).min()
    return dict(total=total, cagr=cagr, sharpe=sharpe, max_dd=max_dd, trades=trades)


def evaluate_ticker(close, cost, split=0.7):
    cut = int(len(close) * split)
    train, test = close.iloc[:cut], close.iloc[cut:]

    best, best_sharpe = None, -np.inf
    for fast in FAST_GRID:
        for slow in SLOW_GRID:
            if fast >= slow:
                continue
            r, n = strategy_returns(train, fast, slow, cost)
            s = metrics(r, n)["sharpe"]
            if pd.notna(s) and s > best_sharpe:
                best, best_sharpe = (fast, slow), s
    if best is None:
        raise ValueError("not enough history for the SMA grid; use more years")

    # Warm up the indicators on train data, then score only the test window.
    full_r, _ = strategy_returns(close, *best, cost)
    test_r = full_r.loc[test.index]
    n_test = int(
        (((close.rolling(best[0]).mean() > close.rolling(best[1]).mean()).astype(float)
          .shift(1).fillna(0.0).diff().abs()).loc[test.index]).sum()
    )
    strat = metrics(test_r, n_test)
    hold = metrics(test.pct_change().fillna(0.0))
    return best, best_sharpe, strat, hold


def pct(x):
    return "n/a" if pd.isna(x) else f"{x * 100:6.1f}%"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", nargs="+", default=DEFAULT_TICKERS)
    ap.add_argument("--years", type=int, default=8)
    ap.add_argument("--cost", type=float, default=0.001, help="cost per position change (0.001 = 0.1%%)")
    ap.add_argument("--out", default="backtest_results.csv")
    args = ap.parse_args()

    rows = []
    for t in args.tickers:
        try:
            close = load_prices(t, args.years)
            (fast, slow), train_sharpe, strat, hold = evaluate_ticker(close, args.cost)
        except Exception as e:  # keep going if one ticker fails
            print(f"{t}: skipped ({e})")
            continue
        rows.append(
            dict(
                ticker=t, fast=fast, slow=slow, train_sharpe=round(train_sharpe, 2),
                test_strat_return=strat["total"], test_hold_return=hold["total"],
                test_strat_sharpe=strat["sharpe"], test_hold_sharpe=hold["sharpe"],
                test_strat_maxdd=strat["max_dd"], test_hold_maxdd=hold["max_dd"],
                test_trades=strat["trades"],
            )
        )

    if not rows:
        print("No results. Check your internet connection and ticker symbols.")
        return

    df = pd.DataFrame(rows)
    print(f"\nOut-of-sample results (last 30% of history, cost {args.cost:.2%} per trade)\n")
    print(f"{'Ticker':<7}{'SMA':<9}{'Strat ret':>10}{'Hold ret':>10}{'Strat Sh':>10}{'Hold Sh':>9}{'Strat DD':>10}{'Trades':>8}")
    for r in rows:
        print(
            f"{r['ticker']:<7}{str(r['fast']) + '/' + str(r['slow']):<9}"
            f"{pct(r['test_strat_return']):>10}{pct(r['test_hold_return']):>10}"
            f"{r['test_strat_sharpe']:>10.2f}{r['test_hold_sharpe']:>9.2f}"
            f"{pct(r['test_strat_maxdd']):>10}{r['test_trades']:>8}"
        )
    beat = (df["test_strat_sharpe"] > df["test_hold_sharpe"]).sum()
    print(f"\nStrategy beat buy-and-hold on risk-adjusted return (Sharpe) in {beat} of {len(df)} tickers.")
    df.to_csv(args.out, index=False)
    print(f"Saved details to {args.out}")


if __name__ == "__main__":
    main()
