"""
Daily review (fake money only). Reads the picker's saved trades and writes review.json
with a plain-English look at what worked and what did not. It only REPORTS; it never
changes the trading rules by itself (changing rules from a handful of trades just fits noise).

    python review.py
"""
import json
from datetime import date

MIN_TRADES_FOR_ADVICE = 30   # closed trades needed before the review suggests rule changes
WHIPSAW_DAYS = 5             # buying a stock back within this many days of selling it


def days_between(a, b):
    return (date.fromisoformat(b) - date.fromisoformat(a)).days


def load(path):
    try:
        return json.load(open(path))
    except Exception:
        return None


def main():
    pick, hold = load("state_picker.json"), load("state_hold.json")
    if not pick:
        print("No picker data yet.")
        return
    open_lots, closed, last_sold, whips = {}, [], {}, []
    for t in pick["trades"]:
        k = t["ticker"]
        if t["side"] == "BUY":
            if k in last_sold and days_between(last_sold[k]["date"], t["date"]) <= WHIPSAW_DAYS:
                whips.append({"ticker": k, "sold": last_sold[k]["date"], "rebought": t["date"],
                              "sold_at": last_sold[k]["price"], "rebought_at": t["price"]})
            open_lots[k] = t
        else:
            b = open_lots.pop(k, None)
            if b:
                closed.append({"ticker": k, "bought": b["date"], "sold": t["date"],
                               "pnl_pct": round((t["price"] / b["price"] - 1) * 100, 1),
                               "days": days_between(b["date"], t["date"]), "why": t.get("why", "")})
            last_sold[k] = t
    px = pick.get("prices", {})
    open_now = [{"ticker": k, "pnl_pct": round((px[k] / b["price"] - 1) * 100, 1)}
                for k, b in open_lots.items() if k in px]

    by_rule = {}
    for c in closed:
        by_rule.setdefault(c["why"], []).append(c["pnl_pct"])
    rules = [{"rule": r, "count": len(v), "avg_pnl_pct": round(sum(v) / len(v), 1)} for r, v in by_rule.items()]

    def ret(s):
        return (s["equity"][-1][1] / 10000 - 1) * 100 if s and s["equity"] else 0.0
    rp, rh = ret(pick), ret(hold)
    wins = sum(1 for c in closed if c["pnl_pct"] > 0)

    lines = [f"The picker is {rp:+.1f}% since the start; the original 10 are {rh:+.1f}%."]
    if closed:
        lines.append(f"{len(closed)} trades closed so far, {wins} made money.")
    for r in rules:
        lines.append(f"Sells for \"{r['rule']}\": {r['count']} so far, average {r['avg_pnl_pct']:+.1f}%.")
    if whips:
        lines.append(f"{len(whips)} stock(s) were bought back within {WHIPSAW_DAYS} days of selling "
                     f"({', '.join(w['ticker'] for w in whips)}). That pays trading costs twice and chases the same bumpy price.")
    losers = [o for o in open_now if o["pnl_pct"] <= -10]
    if losers:
        lines.append("Open positions down 10% or more: " + ", ".join(f"{o['ticker']} {o['pnl_pct']:+.1f}%" for o in losers) + ".")
    if len(closed) < MIN_TRADES_FOR_ADVICE:
        lines.append(f"Only {len(closed)} closed trades, so this is too little to change rules on. "
                     f"The review waits for {MIN_TRADES_FOR_ADVICE}.")
    json.dump({"date": pick["last_date"], "closed": len(closed), "wins": wins, "rules": rules,
               "whipsaws": whips, "open": open_now, "lines": lines,
               "enough_data": len(closed) >= MIN_TRADES_FOR_ADVICE}, open("review.json", "w"), indent=1)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
