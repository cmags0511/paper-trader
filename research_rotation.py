"""
Round 3 research (fake money, past prices): an ACTIVE rotation picker.

Instead of selling when a stock "turns bad", it never sells for that reason. It only sells a
stock to make room for a better-ranked one, and it can hold more than 10 stocks.
Same honest limits as research.py (today's S&P 500 list, strong market, scored on two halves).

    python research_rotation.py [--demo]
"""
import json
import sys

import numpy as np
import pandas as pd

import sp500_picker as sp
from research import COST, WARM, score


def rotation_sim(close, vol, p):
    n, margin, every = p["n"], p.get("margin", 0.0), p.get("every", 1)
    look, skip = p.get("look", 126), p.get("skip", 21)
    plain = p.get("rank", "plain") == "plain"
    need_up = p.get("need_up", True)
    max_swaps = p.get("max_swaps", 99)
    mom = close.shift(skip) / close.shift(look) - 1
    risk = close.pct_change().rolling(60).std() * np.sqrt(252)
    raw = mom if plain else mom / risk.clip(lower=0.1)
    dvol = (close * vol).rolling(20).mean()
    cash, sh, eq, swaps = 10000.0, {}, [], 0
    for i in range(WARM, len(close)):
        px = close.iloc[i]
        s = raw.iloc[i]
        elig = (px > sp.MIN_PRICE) & (dvol.iloc[i] > sp.MIN_DOLLAR_VOL) & s.notna()
        if need_up:
            elig &= mom.iloc[i] > 0
        if i % every == 0:
            cand = [t for t in s[elig].sort_values(ascending=False).index if t not in sh]
            # swap worst holding for a clearly better candidate (never because it "turned bad")
            done = 0
            while cand and sh and done < max_swaps:
                worst = min(sh, key=lambda t: s.get(t, -9) if np.isfinite(s.get(t, -9)) else -9)
                ws = s.get(worst, -9)
                ws = ws if np.isfinite(ws) else -9
                if s[cand[0]] > ws + margin and np.isfinite(px[worst]):
                    cash += sh.pop(worst) * px[worst] * (1 - COST)
                    done += 1
                    swaps += 1
                    t = cand.pop(0)
                    tot_slots = n - len(sh)
                    spend = cash / (1 + COST) / max(tot_slots, 1)
                    sh[t] = spend / px[t]
                    cash -= spend * (1 + COST)
                else:
                    break
            # fill empty slots
            free = n - len(sh)
            if free > 0 and cand:
                spend_each = cash / (1 + COST) / free
                for t in cand[:free]:
                    sh[t] = spend_each / px[t]
                    cash -= spend_each * (1 + COST)
        eq.append(cash + sum(q * px[t] for t, q in sh.items() if np.isfinite(px[t])))
    out = pd.Series(eq, index=close.index[WARM:])
    out.attrs["swaps"] = swaps
    return out


def main():
    demo = "--demo" in sys.argv
    if demo:
        close, vol = sp.demo_data(n=120, days=900)
    else:
        close, vol = sp.download(sp.get_sp500(), 6)
    res = {"period": [str(close.index[WARM].date()), str(close.index[-1].date())], "rotation": {}}
    variants = {
        "10 stocks, swap on any better rank": dict(n=10),
        "20 stocks, swap on any better rank": dict(n=20),
        "30 stocks, swap on any better rank": dict(n=30),
        "50 stocks, swap on any better rank": dict(n=50),
        "10 stocks, swap only if 10 pts better": dict(n=10, margin=0.10),
        "20 stocks, swap only if 10 pts better": dict(n=20, margin=0.10),
        "30 stocks, swap only if 10 pts better": dict(n=30, margin=0.10),
        "20 stocks, swap only if 25 pts better": dict(n=20, margin=0.25),
        "20 stocks, weekly swaps": dict(n=20, every=5),
        "20 stocks, 3-month gain": dict(n=20, look=63),
        "20 stocks, 12-month gain": dict(n=20, look=252, skip=21),
        "20 stocks, gain per bumpiness": dict(n=20, rank="risk"),
        "20 stocks, allow stocks that are falling": dict(n=20, need_up=False),
        "20 stocks, max 2 swaps a day": dict(n=20, max_swaps=2),
        "30 stocks, max 2 swaps a day": dict(n=30, max_swaps=2),
        "30 stocks, 12-month gain, 10 pts": dict(n=30, look=252, margin=0.10),
    }
    for name, p in variants.items():
        eq = rotation_sim(close, vol, p)
        s = score(eq)
        res["rotation"][name] = {"params": p, "swaps": eq.attrs["swaps"], **s}
        print(f"{name:45s} all {s['all']['per_year']:6.1f}%/yr  1st {s['first_half']['per_year']:6.1f}  "
              f"2nd {s['second_half']['per_year']:6.1f}  drop {s['all']['worst_drop']:6.1f}  swaps {eq.attrs['swaps']}")
    if not demo:
        json.dump(res, open("research_rotation.json", "w"), indent=1)


if __name__ == "__main__":
    main()
