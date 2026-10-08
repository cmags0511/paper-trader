# paper-trader

Practice stock trading with **fake money**. Nothing here connects to a broker or can place a real trade.

## What's in here

| File | What it does |
| --- | --- |
| `paper_trader.py` | A practice trading bot. Starts with $10,000 of pretend cash, buys and sells by a simple rule, and shows how it did compared with just holding the stocks. |
| `sp500_picker.py` | Scans the S&P 500 daily and holds the 10 best-ranked stocks. |
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

Try the S&P 500 picker on made-up prices:

    python sp500_picker.py replay --demo

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

The file `.github/workflows/daily.yml` tells GitHub to run two practice portfolios every weekday after the market closes. Each starts with $10,000 of fake cash:

1. **Original 10 (buy and hold)**: buys Apple, Microsoft, Nvidia, Amazon, Google, Meta, JPMorgan, Goldman Sachs, Exxon and Costco once and never sells. Saved in `state_hold.json`.
2. **S&P 500 picker** (`sp500_picker.py`): looks at all ~500 S&P 500 stocks every day and holds only 10 at a time. It only considers stocks over $10 that trade at least $20 million a day and sit above their 100-day average, and ranks them by 6-month gain divided by bumpiness. It sells a stock if it falls 15% from its high, drops below its 100-day average, or falls out of the top 40. Empty slots are filled from the top 20. Saved in `state_picker.json`.

The picker's rules are a simple experiment, not a proven winner. Earlier tests found simple rules rarely beat holding. The stock list is today's S&P 500, so the picker cannot be fairly back-tested on old data.

To start it: open the repo on github.com, click the **Actions** tab, pick **Daily paper trade and website**, and click **Run workflow**.

## The dashboard website

`index.html` shows both portfolios with charts, the picker's top-ranked stocks, and every trade with the reason. It refreshes itself every few minutes.

One-time setup on github.com:
1. Settings, then **Pages**, then under "Build and deployment" set **Source** to **GitHub Actions**.
2. GitHub Pages is free for **public** repos only. The portfolios are fake money, so public is harmless. Settings, **General**, bottom, **Change visibility**.
3. Run **Daily paper trade and website** once. The link to your site appears at the end of the run.
