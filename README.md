# paper-trader

Practice stock trading with **fake money**. Nothing here connects to a broker or can place a real trade.

## What's in here

| File | What it does |
| --- | --- |
| `paper_trader.py` | A practice trading bot. Starts with $10,000 of pretend cash, buys and sells by a simple rule, and shows how it did compared with just holding the stocks. |
| `backtest_sma.py` | Tests one trading idea (moving averages) on past prices and compares it with just holding. |

## Set up (one time)

1. Install Python from https://www.python.org/downloads (on Windows, tick "Add Python to PATH").
2. Open a terminal (Windows: Start, type `cmd`. Mac: Cmd+Space, type `Terminal`).
3. Go to this folder, for example `cd Documents/paper-trader`.
4. Run: `pip install -r requirements.txt`

## Try it

Practice run with made-up prices (no internet needed):

    python paper_trader.py replay --demo

Replay real history day by day:

    python paper_trader.py replay --tickers AAPL MSFT NVDA JPM --years 3 --strategy voltarget

Keep a live practice portfolio (run once after each market close):

    python paper_trader.py daily --tickers AAPL MSFT NVDA JPM --strategy voltarget

Test the moving-average idea on past prices:

    python backtest_sma.py

On a Mac you may need `python3` instead of `python`.

## The strategies

- `hold`: own everything equally and never sell. This is the benchmark to beat.
- `voltarget`: own less of a stock when its price is swinging a lot.
- `sma`: own a stock only while its 50-day average is above its 200-day average.
- `trend200`: own a stock only while its price is above its 200-day average.

## What the research found

In tests on 484 S&P 500 stocks over about 3 years, none of these rules beat simply holding on a risk-adjusted basis. `voltarget` had smaller drops but lower gains. Treat this as a tool for learning and practice, not a way to make money.

## Limits

Prices come from Yahoo Finance through the `yfinance` package, so they can be delayed or occasionally wrong. Results ignore taxes. Past results do not predict future results. This is not financial advice.

## Running automatically every day (real prices)

The file `.github/workflows/daily.yml` tells GitHub to run two practice portfolios every weekday after the market closes: one using `voltarget`, one plain `hold` as the benchmark. Each starts with $10,000 of fake cash. Their holdings are saved in `state_voltarget.json` and `state_hold.json`, which update in this repo after each run.

To start it: open the repo on github.com, click the **Actions** tab, enable workflows if asked, pick **Daily paper trade**, and click **Run workflow** to try it right away. To change the stocks, edit the `TICKERS` line in the workflow file.
