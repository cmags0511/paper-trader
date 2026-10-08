"""
Paper trading bot (simulation only, FAKE money, no broker connection).

It cannot place real trades. It keeps a pretend portfolio, decides what to buy
and sell using a simple rule, and records everything so you can see how it does.

Setup (on your own computer):
    pip install yfinance pandas numpy

Try it right away with made-up practice prices (no internet needed):
    python paper_trader.py replay --demo

Replay real history day by day (starts with $10,000 of fake cash):
    python paper_trader.py replay --tickers AAPL MSFT NVDA JPM --years 3 --strategy voltarget

Run it once per trading day to keep a live paper portfolio going:
    python paper_trader.py daily --tickers AAPL MSFT NVDA JPM --strategy voltarget
    (it saves its state in paper_state.json and tells you the trades it WOULD make)

Strategies:  hold | voltarget | sma | trend200
  hold       own everything equally, never sell (the benchmark to beat)
  voltarget  own less of a stock when it is swinging wildly (best risk-adjusted
             result in my tests, but it did not beat plain holding)
  sma        own a stock only while its 50-day average is above its 200-day
  trend200   own a stock only while its price is above its 200-day average
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

COST = 0.001          # 0.1% of the value of every trade
MIN_TRADE = 0.01      # ignore rebalances smaller than 1% of the portfolio
STATE_FILE = "paper_state.json"


# ---------- data ----------
def load_prices(tickers, years, demo):
    if demo:
        rng = np.random.default_rng(7)
        idx = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=int(years * 252))
        data = {}
        for t in tickers:
            drift, vol = rng.uniform(0.0002, 0.0007), rng.uniform(0.012, 0.03)
            data[t] = 100 * np.exp(np.cumsum(rng.normal(drift, vol, len(idx))))
        return pd.DataFrame(data, index=idx)
    import yfinance as yf

    df = yf.download(tickers, period=f"{years}y", auto_adjust=True, progress=False)["Close"]
    if isinstance(df, pd.Series):
        df = df.to_frame(tickers[0])
    return df.ffill().dropna(how="all")


# ---------- strategies: target weight (0 to 1) for each stock each day ----------
def target_weights(prices, strategy):
    if strategy == "hold":
        w = pd.DataFrame(1.0, index=prices.index, columns=prices.columns)
    elif strategy == "voltarget":
        vol = prices.pct_change().rolling(20).std() * np.sqrt(252)
        w = (0.25 / vol).clip(upper=1.0)
    elif strategy == "sma":
        w = (prices.rolling(50).mean() > prices.rolling(200).mean()).astype(float)
    elif strategy == "trend200":
        w = (prices > prices.rolling(200).mean()).astype(float)
    else:
        raise SystemExit(f"unknown strategy: {strategy}")
    return w.fillna(0.0)


# ---------- the simulated account ----------
def new_state(cash):
    return {"cash": cash, "shares": {}, "last_date": None, "trades": [], "equity": []}


def equity_of(state, px):
    return state["cash"] + sum(sh * px[t] for t, sh in state["shares"].items())


def rebalance(state, date, px, weights):
    """Move the fake portfolio toward its target weights at today's prices."""
    eq = equity_of(state, px)
    n = len(px)
    wants = {t: weights[t] * eq / n for t in px.index}
    orders = []
    for t in px.index:
        have = state["shares"].get(t, 0.0) * px[t]
        orders.append((t, wants[t] - have))
    # sells first so there is cash for the buys
    for t, diff in sorted(orders, key=lambda x: x[1]):
        if abs(diff) < MIN_TRADE * eq or px[t] <= 0:
            continue
        if diff < 0:
            qty = min(-diff / px[t], state["shares"].get(t, 0.0))
            state["shares"][t] = state["shares"].get(t, 0.0) - qty
            state["cash"] += qty * px[t] * (1 - COST)
            side = "SELL"
        else:
            spend = min(diff, state["cash"] / (1 + COST))
            if spend < MIN_TRADE * eq:
                continue
            qty = spend / px[t]
            state["shares"][t] = state["shares"].get(t, 0.0) + qty
            state["cash"] -= spend * (1 + COST)
            side = "BUY"
        state["trades"].append({"date": str(date.date()), "side": side, "ticker": t,
                                "shares": round(qty, 4), "price": round(float(px[t]), 2)})
    state["last_date"] = str(date.date())
    state["equity"].append([str(date.date()), round(float(equity_of(state, px)), 2)])


# ---------- reporting ----------
def report(state, prices, start_idx, cash0):
    eq = pd.Series({d: v for d, v in state["equity"]})
    eq.index = pd.to_datetime(eq.index)
    rets = eq.pct_change().dropna()
    total = eq.iloc[-1] / cash0 - 1
    dd = (eq / eq.cummax() - 1).min()
    sharpe = rets.mean() / rets.std() * np.sqrt(252) if rets.std() > 0 else float("nan")
    held = prices.iloc[start_idx:]
    hold = (1 + held.pct_change().fillna(0).mean(axis=1)).cumprod()  # equal mix, kept even
    hold_total = hold.iloc[-1] - 1
    hold_dd = (hold / hold.cummax() - 1).min()
    print("\n=== Paper trading results (fake money) ===")
    print(f"Start: ${cash0:,.0f}   End: ${eq.iloc[-1]:,.2f}   Days: {len(eq)}")
    print(f"Bot return:          {total:7.1%}   worst drop {dd:7.1%}   Sharpe {sharpe:5.2f}")
    print(f"Just holding (same stocks, equal mix, no costs): {hold_total:7.1%}   worst drop {hold_dd:7.1%}")
    print(f"Trades made: {len(state['trades'])}")
    return eq


def main():
    ap = argparse.ArgumentParser(description="Paper trading bot (fake money only)")
    ap.add_argument("mode", choices=["replay", "daily"])
    ap.add_argument("--tickers", nargs="+", default=["AAPL", "MSFT", "NVDA", "JPM", "XOM", "COST"])
    ap.add_argument("--strategy", default="voltarget", choices=["hold", "voltarget", "sma", "trend200"])
    ap.add_argument("--years", type=float, default=3)
    ap.add_argument("--cash", type=float, default=10000)
    ap.add_argument("--demo", action="store_true", help="use made-up prices, no internet")
    args = ap.parse_args()

    # 200 extra days so the 200-day averages are ready on day one
    prices = load_prices(args.tickers, args.years + 1, args.demo).dropna(axis=1, how="all")
    weights = target_weights(prices, args.strategy)

    if args.mode == "replay":
        start = min(200, len(prices) - 2)
        state = new_state(args.cash)
        for i in range(start, len(prices)):
            rebalance(state, prices.index[i], prices.iloc[i], weights.iloc[i])
        eq = report(state, prices, start, args.cash)
        pd.DataFrame(state["trades"]).to_csv("paper_trades.csv", index=False)
        eq.to_csv("paper_equity.csv", header=["equity"])
        print("Saved paper_trades.csv and paper_equity.csv")
    else:  # daily
        state = json.load(open(STATE_FILE)) if os.path.exists(STATE_FILE) else new_state(args.cash)
        date, px, w = prices.index[-1], prices.iloc[-1], weights.iloc[-1]
        if state["last_date"] == str(date.date()):
            print(f"Already ran for {date.date()}. Run again after the next market close.")
        else:
            before = len(state["trades"])
            rebalance(state, date, px, w)
            print(f"Paper orders for {date.date()} (NOT real, nothing was sent anywhere):")
            for tr in state["trades"][before:]:
                print(f"  {tr['side']:<4} {tr['shares']:>10} {tr['ticker']} @ ${tr['price']}")
            if len(state["trades"]) == before:
                print("  none today")
            json.dump(state, open(STATE_FILE, "w"), indent=1)
        eqv = equity_of(state, px)
        print(f"Paper portfolio value: ${eqv:,.2f}  (cash ${state['cash']:,.2f})")
        for t, sh in state["shares"].items():
            if sh > 1e-9:
                print(f"  {t}: {sh:.4f} shares = ${sh * px[t]:,.2f}")


if __name__ == "__main__":
    main()
