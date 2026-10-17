"""
Strategy research (fake money, past prices only). Tests many versions of both methods on
~5 years of real prices and writes research_results.json.

Honest limits: it uses TODAY's S&P 500 list for the past (survivorship bias, flatters stock
pickers), the last 5 years were mostly a rising market, and with many versions tested some
will look good by luck. So every version is scored on two separate halves of the period and
we only trust ones that do well in BOTH.

    python research.py            (needs internet; run by GitHub Actions)
    python research.py --demo     (made-up prices, just checks the code runs)
"""
import json
import sys

import numpy as np
import pandas as pd

import sp500_picker as sp

COST = 0.001
HOLD10 = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "JPM", "GS", "XOM", "COST"]
WARM = 260

BASE = dict(ma=100, mom_skip=21, mom_look=126, rank="mom_risk", stop=0.15, keep=40, pool=20,
            cooldown=0, every=1, regime=False, buffer=0.0, slots=10, minp=10.0)


def picker_sim(close, vol, spy, p):
    p = {**BASE, **p}
    ma = close.rolling(p["ma"]).mean()
    mom = close.shift(p["mom_skip"]) / close.shift(p["mom_look"]) - 1
    risk = close.pct_change().rolling(60).std() * np.sqrt(252)
    dvol = (close * vol).rolling(20).mean()
    spy_ok = (spy > spy.rolling(200).mean()) if p["regime"] else pd.Series(True, index=close.index)
    cash, sh, peak, sold_on, eq = 10000.0, {}, {}, {}, []
    cols = close.columns
    for i in range(WARM, len(close)):
        px = close.iloc[i]
        ok = (px > p["minp"]) & (px > ma.iloc[i] * (1 + p["buffer"])) & (dvol.iloc[i] > sp.MIN_DOLLAR_VOL) & (mom.iloc[i] > 0)
        if p["rank"] == "mom_risk":
            sc = mom.iloc[i] / risk.iloc[i].clip(lower=0.1)
        elif p["rank"] == "mom":
            sc = mom.iloc[i]
        else:  # lowvol among positive-momentum stocks
            sc = -risk.iloc[i]
        ranked = list(sc[ok].dropna().sort_values(ascending=False).index)
        rank = {t: r + 1 for r, t in enumerate(ranked)}
        market_ok = bool(spy_ok.iloc[i])
        for t in list(sh):
            q = px[t]
            if not np.isfinite(q):
                continue
            peak[t] = max(peak.get(t, q), q)
            bad = (p["stop"] and q < peak[t] * (1 - p["stop"]))
            slow = (i % p["every"] == 0) and (q < ma[t].iloc[i] * (1 - p["buffer"]) or rank.get(t, 999) > p["keep"])
            if bad or slow or not market_ok:
                cash += sh.pop(t) * q * (1 - COST)
                sold_on[t] = i
        if market_ok and i % p["every"] == 0:
            free = p["slots"] - len(sh)
            if free > 0:
                tot = cash + sum(n * px[t] for t, n in sh.items() if np.isfinite(px[t]))
                target = tot / p["slots"]
                for t in ranked[:p["pool"]]:
                    if free <= 0:
                        break
                    if t in sh or i - sold_on.get(t, -999) < p["cooldown"]:
                        continue
                    spend = min(target, cash / (1 + COST))
                    if spend < 0.5 * target:
                        break
                    sh[t] = spend / px[t]
                    peak[t] = px[t]
                    cash -= spend * (1 + COST)
                    free -= 1
        eq.append(cash + sum(n * px[t] for t, n in sh.items() if np.isfinite(px[t])))
    return pd.Series(eq, index=close.index[WARM:])


def hold_sim(close, tickers, mode):
    px = close[tickers].iloc[WARM:]
    n = len(tickers)
    if mode == "buy_once":
        return 10000 * (1 - COST) * (px / px.iloc[0]).mean(axis=1)
    ret = px.pct_change().fillna(0)
    if mode == "monthly_rebalance":
        w = pd.Series(1 / n, index=tickers)
        eq, v = [], 10000.0
        for k, (d, r) in enumerate(ret.iterrows()):
            v *= 1 + (w * r).sum()
            w = w * (1 + r) / (1 + (w * r).sum())
            if k % 21 == 0:
                v *= 1 - COST * (w - 1 / n).abs().sum()
                w = pd.Series(1 / n, index=tickers)
            eq.append(v)
        return pd.Series(eq, index=px.index)
    ma = close[tickers].rolling(200).mean().iloc[WARM:]
    on = (px > ma).shift(1).fillna(False).astype(float)   # decide on yesterday's close
    flips = on.diff().abs().fillna(0).sum(axis=1)
    daily = (ret * on).sum(axis=1) / n
    daily = daily - flips * COST / n
    return 10000 * (1 + daily).cumprod()


def stats(eq):
    r = eq.pct_change().dropna()
    yrs = len(r) / 252
    return dict(ret=round(float(eq.iloc[-1] / eq.iloc[0] - 1) * 100, 1),
                per_year=round(float((eq.iloc[-1] / eq.iloc[0]) ** (1 / yrs) - 1) * 100, 1),
                worst_drop=round(float((eq / eq.cummax() - 1).min()) * 100, 1),
                sharpe=round(float(r.mean() / r.std() * np.sqrt(252)), 2) if r.std() > 0 else 0)


def score(eq):
    mid = len(eq) // 2
    a, b = eq.iloc[:mid], eq.iloc[mid:]
    return dict(all=stats(eq), first_half=stats(a), second_half=stats(b))


def main():
    demo = "--demo" in sys.argv
    if demo:
        close, vol = sp.demo_data(n=120, days=900)
        close["SPY"] = close.iloc[:, 0]
        vol["SPY"] = vol.iloc[:, 0]
        hold_names = list(close.columns[:10])
    else:
        close, vol = sp.download(sp.get_sp500() + ["SPY"], 6)
        hold_names = [t for t in HOLD10 if t in close.columns]
    spy = close["SPY"]
    stocks = [c for c in close.columns if c != "SPY"]

    out = {"period": [str(close.index[WARM].date()), str(close.index[-1].date())], "picker": {}, "hold10": {}, "benchmarks": {}}

    out["benchmarks"]["S&P 500 index fund (SPY)"] = score(10000 * spy.iloc[WARM:] / spy.iloc[WARM])
    ew = (1 + close[stocks].pct_change().fillna(0).mean(axis=1)).cumprod().iloc[WARM:]
    out["benchmarks"]["All 500 stocks equal weight"] = score(10000 * ew / ew.iloc[0])

    for mode in ["buy_once", "monthly_rebalance", "trend_filter_200d"]:
        out["hold10"][mode] = score(hold_sim(close, hold_names, mode))

    variants = {
        "baseline (current live rules)": {},
        "no stop-loss": dict(stop=None),
        "stop 25%": dict(stop=0.25),
        "stop 10%": dict(stop=0.10),
        "wait 20 days before buying a sold stock back": dict(cooldown=20),
        "wait 40 days before buying back": dict(cooldown=40),
        "trade only once a week": dict(every=5),
        "trade only once a month": dict(every=21),
        "200-day average instead of 100": dict(ma=200),
        "50-day average": dict(ma=50),
        "sell only 3% below the average": dict(buffer=0.03),
        "rank by plain 6-month gain": dict(rank="mom"),
        "12-month gain (skip last month)": dict(mom_look=252),
        "3-month gain": dict(mom_look=63),
        "rank by low bumpiness": dict(rank="lowvol"),
        "keep while in top 100": dict(keep=100),
        "cash when the S&P 500 is below its 200-day average": dict(regime=True),
        "weekly + cooldown 20 + stop 20%": dict(every=5, cooldown=20, stop=0.20),
        "monthly + 200-day + market filter": dict(every=21, ma=200, regime=True),
        "weekly + 12-month gain + market filter": dict(every=5, mom_look=252, regime=True),
    }
    for name, params in variants.items():
        out["picker"][name] = {"params": params, **score(picker_sim(close[stocks], vol[stocks], spy, params))}
        s = out["picker"][name]
        print(f"{name:55s} all {s['all']['per_year']:6.1f}%/yr  1st {s['first_half']['per_year']:6.1f}  2nd {s['second_half']['per_year']:6.1f}  drop {s['all']['worst_drop']:6.1f}")
    for sect in ("hold10", "benchmarks"):
        for k, s in out[sect].items():
            print(f"{k:55s} all {s['all']['per_year']:6.1f}%/yr  1st {s['first_half']['per_year']:6.1f}  2nd {s['second_half']['per_year']:6.1f}  drop {s['all']['worst_drop']:6.1f}")
    if not demo:
        json.dump(out, open("research_results.json", "w"), indent=1)


if __name__ == "__main__":
    main()
