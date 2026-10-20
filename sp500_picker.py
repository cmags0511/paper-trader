"""
S&P 500 picker (simulation only, FAKE money, no broker connection).

Every trading day it looks at all ~500 S&P 500 stocks, ranks them, and keeps
at most 10 positions. It sells stocks that have gone bad and buys the best
ranked stocks to fill the empty slots.

Ranking (a "realistic" filter, no penny-stock or thin-trading surprises):
  - only stocks priced above $10 that trade at least $20 million a day
  - only stocks whose price is above their 100-day average (an uptrend)
  - score = the last 6 months' gain (skipping the most recent month), divided by
    how bumpy the stock has been, so smooth steady risers rank above wild ones

Selling rules:
  - the price falls 15% from its highest point since we bought it
  - the price drops below its 100-day average
  - the stock falls out of the top 40 of the ranking

    python sp500_picker.py replay --demo
    python sp500_picker.py daily --state state_picker.json
"""
import argparse
import io
import json
import os
import time
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from paper_trader import COST, equity_of, new_state

SLOTS = 10
STOP = 0.15
MIN_PRICE = 10.0
MIN_DOLLAR_VOL = 20e6
BUY_POOL = 20
KEEP_RANK = 40
MA_DAYS = 100         # "trend" average length
RANK_MODE = "risk"    # "risk": gain / bumpiness   "plain": gain only
LIST_FILE = "sp500.txt"
STATE_FILE = "state_picker.json"


# ---------- data ----------
def get_sp500():
    """Today's S&P 500 list from Wikipedia, cached in sp500.txt."""
    try:
        import requests
        r = requests.get("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
                         headers={"User-Agent": "paper-trader/1.0"}, timeout=30)
        r.raise_for_status()
        tick = pd.read_html(io.StringIO(r.text))[0]["Symbol"].astype(str)
        tick = sorted({t.strip().replace(".", "-") for t in tick})
        if len(tick) >= 400:
            open(LIST_FILE, "w").write("\n".join(tick))
            return tick
    except Exception as e:  # fall back to the saved list
        print("Could not fetch the S&P 500 list:", e)
    if os.path.exists(LIST_FILE):
        return open(LIST_FILE).read().split()
    raise SystemExit("No S&P 500 list available.")


def download(tickers, years):
    import yfinance as yf
    closes, vols = [], []
    for i in range(0, len(tickers), 100):
        chunk = tickers[i:i + 100]
        for attempt in range(3):
            try:
                df = yf.download(chunk, period=f"{years}y", auto_adjust=True,
                                 progress=False, threads=True)
                if len(df):
                    closes.append(df["Close"])
                    vols.append(df["Volume"])
                    break
            except Exception as e:
                print("retry", e)
            time.sleep(5)
    if not closes:
        raise SystemExit("No price data came back.")
    close = pd.concat(closes, axis=1).sort_index().ffill().dropna(how="all")
    vol = pd.concat(vols, axis=1).sort_index().reindex(close.index)
    close = close.dropna(axis=1, how="all")
    if close.shape[1] < 400:
        raise SystemExit(f"Only got {close.shape[1]} stocks; refusing to trade on partial data.")
    return close, vol.reindex(columns=close.columns)


def demo_data(n=120, days=700):
    rng = np.random.default_rng(3)
    idx = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=days)
    close, vol = {}, {}
    for i in range(n):
        t = f"S{i:03d}"
        close[t] = rng.uniform(15, 300) * np.exp(np.cumsum(rng.normal(rng.uniform(0, 0.0008),
                                                                     rng.uniform(0.01, 0.03), days)))
        vol[t] = rng.uniform(1e6, 5e6, days)
    return pd.DataFrame(close, index=idx), pd.DataFrame(vol, index=idx)


# ---------- ranking ----------
def indicators(close, vol):
    ma100 = close.rolling(MA_DAYS).mean()
    mom = close.shift(21) / close.shift(126) - 1
    risk = close.pct_change().rolling(60).std() * np.sqrt(252)
    dvol = (close * vol).rolling(20).mean()
    return ma100, mom, risk, dvol


def scan(i, close, ma100, mom, risk, dvol):
    """Ranked list of buyable stocks on day i (best first)."""
    px = close.iloc[i]
    ok = (px > MIN_PRICE) & (px > ma100.iloc[i]) & (dvol.iloc[i] > MIN_DOLLAR_VOL) & (mom.iloc[i] > 0)
    raw = mom.iloc[i] if RANK_MODE == "plain" else mom.iloc[i] / risk.iloc[i].clip(lower=0.1)
    score = raw[ok].dropna().sort_values(ascending=False)
    return list(score.index), score


# ---------- trading ----------
def sell(state, date, t, p, why):
    sh = state["shares"].pop(t)
    state["cash"] += sh * p * (1 - COST)
    state["peaks"].pop(t, None)
    state["trades"].append({"date": str(date.date()), "side": "SELL", "ticker": t,
                            "shares": round(sh, 4), "price": round(float(p), 2), "why": why})


def step(state, date, i, close, ind):
    ma100, mom, risk, dvol = ind
    px = close.iloc[i]
    state.setdefault("peaks", {})
    ranked, score = scan(i, close, *ind)
    rank = {t: r + 1 for r, t in enumerate(ranked)}
    # 1. sells
    for t in list(state["shares"]):
        p = px.get(t)
        if p is None or not np.isfinite(p):
            continue
        state["peaks"][t] = max(state["peaks"].get(t, p), float(p))
        why = None
        if p < state["peaks"][t] * (1 - STOP):
            why = f"fell {STOP:.0%} from its high"
        elif p < ma100[t].iloc[i]:
            why = f"dropped below its {MA_DAYS}-day average"
        elif rank.get(t, 999) > KEEP_RANK:
            why = "no longer ranks in the top 40"
        if why:
            sell(state, date, t, p, why)
    # 2. buys to fill empty slots
    free = SLOTS - len(state["shares"])
    if free > 0:
        eq = state["cash"] + sum(sh * px[t] for t, sh in state["shares"].items() if np.isfinite(px[t]))
        target = eq / SLOTS
        for t in ranked[:BUY_POOL]:
            if free <= 0:
                break
            if t in state["shares"]:
                continue
            spend = min(target, state["cash"] / (1 + COST))
            if spend < 0.5 * target:
                break
            p = float(px[t])
            state["shares"][t] = spend / p
            state["cash"] -= spend * (1 + COST)
            state["peaks"][t] = p
            state["trades"].append({"date": str(date.date()), "side": "BUY", "ticker": t,
                                    "shares": round(spend / p, 4), "price": round(p, 2),
                                    "why": f"rank #{rank[t]} of {len(ranked)} buyable stocks"})
            free -= 1
    held = {t: float(px[t]) for t in state["shares"] if np.isfinite(px[t])}
    last = {t: float(px[t]) for t in ranked[:15]}
    state["prices"] = {**held, **last}
    state["scan"] = {"date": str(date.date()), "buyable": len(ranked),
                     "top": [{"t": t, "score": round(float(score[t]), 2),
                              "mom": round(float(mom[t].iloc[i]), 3)} for t in ranked[:10]]}
    state["last_date"] = str(date.date())
    eq = state["cash"] + sum(sh * held.get(t, 0.0) for t, sh in state["shares"].items())
    state["equity"].append([str(date.date()), round(float(eq), 2)])


def step_rotate(state, date, i, close, ind, slots, margin):
    """Active rotation: never sells because a stock turned bad. A holding is only sold to
    make room for a clearly better-ranked stock (6-month gain higher by at least `margin`)."""
    ma100, mom, risk, dvol = ind
    px = close.iloc[i]
    s = mom.iloc[i]
    elig = (px > MIN_PRICE) & (dvol.iloc[i] > MIN_DOLLAR_VOL) & s.notna() & (s > 0)
    cand = [t for t in s[elig].sort_values(ascending=False).index if t not in state["shares"]]
    sc = lambda t: float(s[t]) if t in s.index and np.isfinite(s[t]) else -9.0
    state.setdefault("peaks", {})
    while cand and state["shares"]:
        worst = min(state["shares"], key=sc)
        if not (sc(cand[0]) > sc(worst) + margin and np.isfinite(px[worst])):
            break
        new = cand.pop(0)
        sell(state, date, worst, px[worst],
             f"swapped for {new}: its 6-month gain {sc(new):+.0%} vs {sc(worst):+.0%}")
        spend = state["cash"] / (1 + COST) / max(slots - len(state["shares"]), 1)
        buy(state, date, new, float(px[new]), spend, f"replaces {worst}, 6-month gain {sc(new):+.0%}")
    free = slots - len(state["shares"])
    if free > 0 and cand:
        each = state["cash"] / (1 + COST) / free
        for t in cand[:free]:
            buy(state, date, t, float(px[t]), each, f"open slot, 6-month gain {sc(t):+.0%}")
    ranked = [t for t in s[elig].sort_values(ascending=False).index]
    held = {t: float(px[t]) for t in state["shares"] if np.isfinite(px[t])}
    state["prices"] = {**held, **{t: float(px[t]) for t in ranked[:15]}}
    state["scan"] = {"date": str(date.date()), "buyable": len(ranked),
                     "top": [{"t": t, "score": round(sc(t), 2), "mom": round(sc(t), 3)} for t in ranked[:10]]}
    state["last_date"] = str(date.date())
    eq = state["cash"] + sum(sh * held.get(t, 0.0) for t, sh in state["shares"].items())
    state["equity"].append([str(date.date()), round(float(eq), 2)])


def buy(state, date, t, p, spend, why):
    if spend < 1 or not np.isfinite(p):
        return
    state["shares"][t] = state["shares"].get(t, 0.0) + spend / p
    state["cash"] -= spend * (1 + COST)
    state["trades"].append({"date": str(date.date()), "side": "BUY", "ticker": t,
                            "shares": round(spend / p, 4), "price": round(p, 2), "why": why})


def market_open_guard(date):
    now = datetime.now(ZoneInfo("America/New_York"))
    return str(date.date()) == str(now.date()) and (now.hour, now.minute) < (16, 30)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["replay", "daily"])
    ap.add_argument("--state", default=STATE_FILE)
    ap.add_argument("--cash", type=float, default=10000)
    ap.add_argument("--years", type=float, default=2)
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--ma", type=int, default=100, help="trend average length in days")
    ap.add_argument("--rank", choices=["risk", "plain"], default="risk")
    ap.add_argument("--stop", type=float, default=0.15, help="sell after this fall from the high")
    ap.add_argument("--rotate", action="store_true", help="active rotation instead of sell-when-bad")
    ap.add_argument("--slots", type=int, default=20)
    ap.add_argument("--margin", type=float, default=0.25)
    a = ap.parse_args()
    global MA_DAYS, RANK_MODE, STOP
    MA_DAYS, RANK_MODE, STOP = a.ma, a.rank, a.stop

    close, vol = demo_data() if a.demo else download(get_sp500(), a.years + 1)
    ind = indicators(close, vol)
    stepper = (lambda st, d, i, c, n: step_rotate(st, d, i, c, n, a.slots, a.margin)) if a.rotate else step

    if a.mode == "replay":
        state = new_state(a.cash)
        for i in range(130, len(close)):
            stepper(state, close.index[i], i, close, ind)
        e = pd.Series({d: v for d, v in state["equity"]})
        dd = (e / e.cummax() - 1).min()
        print(f"Start ${a.cash:,.0f}  End ${e.iloc[-1]:,.2f}  ({e.iloc[-1] / a.cash - 1:.1%})  "
              f"worst drop {dd:.1%}  trades {len(state['trades'])}  holding {len(state['shares'])}")
        return

    state = json.load(open(a.state)) if os.path.exists(a.state) else new_state(a.cash)
    i = len(close) - 1
    date = close.index[i]
    if not a.demo and market_open_guard(date):
        print("Today's market is not closed yet; skipping.")
        return
    if state["last_date"] == str(date.date()):
        print(f"Already ran for {date.date()}.")
        return
    before = len(state["trades"])
    stepper(state, date, i, close, ind)
    for tr in state["trades"][before:]:
        print(f"  {tr['side']:<4} {tr['ticker']} @ ${tr['price']}  ({tr['why']})")
    json.dump(state, open(a.state, "w"), indent=1)
    print(f"Value ${state['equity'][-1][1]:,.2f}, holding {len(state['shares'])}")


if __name__ == "__main__":
    main()
