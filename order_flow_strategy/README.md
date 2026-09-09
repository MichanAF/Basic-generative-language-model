# Order-Flow Resistance / Two-Candle Reversal Strategy

A self-contained research toolkit that reads order flow to find where supply and
demand actually sit, waits for a two-candle reversal at one of those levels, and
places the trade on the second candle.

Pure Python 3.9+ standard library. No numpy, no pandas, no network access.

> **Research tool, not financial advice.** Nothing here is a prediction, an
> endorsement, or a claim of profitability. The defaults are a starting point,
> not a tuned edge. Read [Honest limitations](#honest-limitations) before you
> risk anything.

---

## Quickstart

```bash
# Generated tape, so you can see the whole pipeline work immediately
python -m order_flow_strategy demo

# Your own tick data
python -m order_flow_strategy backtest --trades ticks.csv --timeframe 300 \
    --tick-size 0.25 --tick-value 12.50

# Two years of BTC: fetch, then run (see "Getting BTC data")
python -m order_flow_strategy.sources.fetch --symbol BTCUSDT --interval 5m \
    --start 2023-08 --end 2025-07 --out data/btc
python -m order_flow_strategy backtest --binance-klines data/btc --preset btc

# What the rules say about the current state of the market
python -m order_flow_strategy scan --trades ticks.csv --timeframe 300

# Walk-forward search, with recommendations and the caveats attached
python -m order_flow_strategy optimize --trades ticks.csv --folds 4

# Compare against the benchmarks it has to beat
python -m order_flow_strategy compare --binance-klines data/btc --preset btc

# ...or write the whole thing out as one HTML page
python -m order_flow_strategy report --binance-klines data/btc --preset btc \
    --out report.html

# Forward-test on a schedule: keeps a book on disk, places no orders
python -m order_flow_strategy paper --binance-klines data/btc --preset btc

# The set-and-forget core: spot, binary exposure, one parameter
python -m order_flow_strategy autopilot --synthetic --days 2000
python -m order_flow_strategy autopilot --binance-klines data/btc-daily --preset btc

# Tests
python -m unittest discover -s order_flow_strategy/tests -t .
```

Library use:

```python
from order_flow_strategy import Backtester, StrategyConfig, bars_from_trades, load_trades_csv
from order_flow_strategy.metrics import summarize
from order_flow_strategy.report import format_stats

bars = bars_from_trades(load_trades_csv("ticks.csv"), 300.0, tick_size=0.25)
result = Backtester(StrategyConfig(tick_size=0.25, tick_value=12.50)).run(bars)
print(format_stats(summarize(result)))
```

---

## The idea

A horizontal line drawn on a chart is a guess about where a large passive seller
is working an order. Order flow lets you stop guessing.

The tell is a **disagreement between aggression and price**. If buyers are
lifting the offer hard and price is not going up, somebody is selling them
everything they want. That seller is the resistance, and their price is the
level worth trading against. Volume alone cannot show this — you need to know
which side was the aggressor on each print.

Once a level is marked, the strategy does not fade the first touch. It waits for
one candle to be rejected and a second candle to confirm.

### Why the second candle

Candle 1 is a hypothesis: price reached the level and closed away from it.
Candle 2 is the market agreeing. Most losses in a naive "fade the level" rule
come from tests that fail immediately — price pokes through, you short the wick,
and the next bar runs you over. Requiring a confirming bar filters exactly that.

The cost is real and worth stating plainly: you enter at a worse price, your
stop is further away, and setups that reverse in a single violent bar leave
without you. The trade-off is only worth it if the second candle removes more
bad trades than good ones — which is what `optimize` is for.

There is no candle 3. If candle 2 does not confirm, the setup is dead.
(`setup_expiry_bars` can widen this if you want to test it; the default is 1.)

---

## Data requirements

This is the part that decides whether any of this works.

| Input | Flag | What you get |
|---|---|---|
| **Tick trades with aggressor side** | `--trades`, `--binance-aggtrades` | Everything. Real footprints, real absorption, real imbalances. |
| OHLCV + reported buy/sell volume | `--bars`, `--binance-klines` | Real delta, estimated distribution across price. Absorption detection is weak. |
| OHLCV alone | `--bars` | A proxy. The buy/sell split is derived *from the close*, so it cannot detect absorption — the one thing this strategy is built on. |

Runs on proxy data are flagged `used_proxy_footprints` and the report says so.
If you see that warning, the backtest is testing a candle pattern, not an
order-flow strategy.

**Trades CSV** (`ts, price, size, side`; aliases such as `timestamp`, `qty`,
`aggressor` are accepted):

```csv
ts,price,size,side
1712000000,4500.25,3,buy
1712000001,4500.00,5,sell
```

`side` accepts `buy`/`sell`, `b`/`s`, `bid`/`ask`, or `1`/`-1`. If your feed has
no aggressor field, most vendors let you reconstruct it with the
tick rule against quotes — do that before using this, not after.

**Bars CSV**: `ts, open, high, low, close, volume`, optionally `ask_volume` and
`bid_volume` (or `delta`), which are used when present.

### Getting BTC data

Binance publishes both archives this needs. Fetch them wherever outbound HTTPS
to `data.binance.vision` is allowed:

```bash
# two years of 5m bars with measured taker-buy volume (~40 MB total)
python -m order_flow_strategy.sources.fetch --symbol BTCUSDT --interval 5m \
    --start 2023-08 --end 2025-07 --out data/btc

python -m order_flow_strategy backtest --binance-klines data/btc --preset btc
```

Klines carry `taker_buy_base_asset_volume`, so **bar delta is measured, not
inferred from the close** — much stronger than plain OHLCV, and small enough to
cover years. For true footprints you need the tape:

```bash
# one month of aggTrades: real aggressor side, but 1-3 GB per month
python -m order_flow_strategy.sources.fetch --symbol BTCUSDT --kind aggTrades \
    --start 2025-06 --end 2025-06 --out data/btc-tape

python -m order_flow_strategy backtest --binance-aggtrades data/btc-tape \
    --preset btc --timeframe 300
```

A practical split: validate the rules across two years of klines, then confirm
on one or two months of aggTrades that the absorption signal is really there.

The readers accept `.zip` as published, `.csv`, or `.csv.gz`, with or without a
header, and handle both the millisecond and microsecond epoch conventions
Binance has used.

### Presets

`--preset btc` and `--preset es` set the instrument conventions; explicit flags
override them. The BTC preset exists because four things differ from a futures
contract, and each one silently invalidates a backtest if left wrong:

- **Row size.** `tick_size` is the footprint row height, *not* the exchange
  tick. BTCUSDT ticks at $0.01; at that row size a 5m bar spanning $300 has
  30,000 rows — unreadable and ~1000x slower. The preset uses $10 rows, giving
  20–40 rows per bar, which is what a real footprint chart shows.
- **Fractional size** — `whole_units` off.
- **Proportional costs** — see below.
- **No session, 24/7** — level half-life stretched to 288 bars (24h).

---

## The rules

### 1. Where resistance sits

Every closed bar is scanned for evidence, and each kind carries a weight:

| Evidence | Weight | What it means |
|---|---|---|
| `absorption` | 2.0 | Heavy aggressive buying into the top of the bar, close near the low. Passive size ate the initiative. |
| `imbalance_stack` | 1.5 | Three or more consecutive price levels dominated by one side on the diagonal comparison. |
| `delta_divergence` | 1.5 | A higher swing high on lower cumulative delta. The push was thinner than the last one. |
| `high_volume_node` | 1.0 | A local peak in the rolling volume profile. Prices the market already agreed on get defended. |
| `wick_rejection` | 0.5 | A long wick with delta to match. Weak on its own; useful as a tiebreaker. |
| `retest_hold` | 1.0 | The level turned price away again. Throttled so chop across a level is one test, not thirty. |

Levels are merged within `level_merge_atr`, decay with a half-life of
`level_half_life` bars, are capped at strength 12, and are invalidated by a
close beyond them of more than `level_break_atr` × ATR. A broken level flips
polarity — broken resistance becomes support at 60% of its strength.

Two gates decide whether a level is tradeable rather than merely marked:

- strength above `level_min_strength`, and
- **at least one genuine order-flow signature** — `absorption`,
  `imbalance_stack`, or `delta_divergence` (`require_flow_backed_levels`).

The second gate matters more than the first. A level built from a wick, a volume
node and a couple of retests can easily clear any strength threshold while
containing no order-flow information at all — that is a chart pattern, and
trading it is not trading order flow. Such levels are still tracked and still
shown by `scan`, they are just not traded. `--allow-chart-levels` turns the gate
off so you can measure what it is worth on your data.

### 2. Candle 1 — the test

For a short at resistance, all of:

- the high reaches within `touch_tol_atr` × ATR of the level;
- the bar **closes back below** the level (a close above is acceptance, not rejection);
- the upper wick is at least `wick_frac` of the bar's range;
- **and at least one order-flow confirmation** (`require_flow_evidence`):
  normalized delta ≤ `c1_delta_max`, or absorption in the top cluster, or
  `imbalance_stack` stacked sell imbalances.

That last requirement is the difference between this and a pin-bar strategy.
Turning it off with `--allow-loose-flow` is supported so you can measure how
much it contributes; expect it to matter.

### 3. Candle 2 — the entry

Fatal (setup is wrong, not early):

- candle 2 takes out candle 1's high, or
- candle 2 closes back above the level.

Required to confirm:

- bearish close, with a body of at least `c2_body_frac_min` of its range;
- close beyond `c2_close_beyond_c1_frac` of candle 1's range, measured down from
  candle 1's high (0.5 = below candle 1's midpoint, 1.0 = below its low);
- normalized delta ≤ `c2_delta_max`;
- optionally weaker than candle 1's delta (`c2_delta_must_worsen`).

Entry goes on immediately, two ways:

- `close_of_2` (default) — market order at candle 2's close. Certain fill, worse price.
- `break_of_2` — resting stop order one tick below candle 2's low, live for
  `entry_valid_bars`. Better price, and it never fills on the setups that
  immediately reverse — at the cost of missing the ones that run without a pullback.

Longs at support mirror all of the above exactly.

### 4. Risk

- **Stop**: beyond `max(level, candle 1 high)` by `stop_buffer_atr` × ATR.
- **Size**: `risk_per_trade` of equity divided by stop distance, rounded down to
  whole units when `whole_units` is on.
- **Target**: `target_r` multiples of risk.
- **Scale-out**: `partial_frac` at `partial_at_r`, then stop to breakeven.
- **Time stop**: flatten after `time_stop_bars` bars.
- **Cooldown**: a level that produced a loss is benched for `cooldown_bars`, and
  no level is traded more than `max_trades_per_level` times.

---

## Recommendations

### Markets and timeframes

Trade this where the order flow is real: a **single centralized order book with
a public tape**. CME futures (ES, NQ, CL, GC, ZN), and major crypto perpetuals
on a single venue. Do not use it on spot FX (no central tape), on CFDs (your
broker's book is not the market), or on thin equities where one participant
moves the whole profile.

**5-minute bars are the sweet spot** for index futures. Below 1 minute the two-
candle wait is mostly noise and costs dominate. Above 15 minutes you get too few
setups to ever reach statistical significance, and the stop distances get large
enough that a single loss hurts.

Trade the liquid part of the session — for US index futures, roughly the RTH
open through the first few hours (`--session-start 810 --session-end 1080`,
minutes from UTC midnight). Overnight levels are worth *marking* but are poor to
*trade*: thin books make absorption look real when it is one participant.

### Which variant to trade

Start with `regime_filter="with_trend_only"`. Selling a pullback into resistance
inside a downtrend is a fundamentally different trade from fading a level inside
a rally, and the first has a much better base rate. Pure counter-trend fading
(`counter_trend_only`) works in balanced, rotational markets and is where this
kind of setup goes to die in a trend day. If you cannot classify the regime
reliably, `none` plus a strict level-strength floor is safer than guessing.

### Parameters worth changing first

In order of how much they matter:

1. **`c2_close_beyond_c1_frac`** — the single biggest lever on the trade-off
   between signal quality and signal count. 0.33 gives many mediocre setups,
   0.75 gives few good ones and much worse entries.
2. **`entry_mode`** — test both on your data. `break_of_2` usually improves
   expectancy per trade and reduces trade count; whether that is a good deal
   depends on your cost structure.
3. **`stop_buffer_atr`** — too tight and you get wicked out by the same
   participant you are trading with; too wide and your size collapses. If
   winners average more than ~0.8 R of heat (the report tells you), it is too tight.
4. **`target_r`** — check the `MFE on losers` line. If losers routinely reach
   1.5 R before failing, a 3 R target is fighting the actual behaviour.
5. **Level weights** (`EvidenceWeights`) — leave these alone until everything
   else is settled. They are the easiest way to overfit.

Do not tune everything at once. Each parameter you add to a grid raises the
chance the best result is noise.

### The cost hurdle — read this before trading BTC

Costs are modelled with a fixed component (per contract, for futures) and a
proportional one (basis points of notional, for crypto). They add, and the
report prints the result as **cost hurdle in R per trade** — the fraction of
your risk that fees and slippage consume before the strategy does anything.

It is the most decision-relevant number in the report, because it is set by
arithmetic rather than by skill:

```
cost_R  ≈  2 × (fee_rate + slippage_rate) × price  ÷  stop_distance
```

Tight stops make it worse, not better. At BTC $60,000 with a 0.5% stop:

| Venue | Taker fee | Cost hurdle |
|---|---|---|
| Binance spot, base tier | 10 bp | **0.48 R per trade** |
| USD-M futures, base tier | 4 bp | 0.24 R |
| USD-M futures, VIP tier | 1.8 bp | 0.15 R |

At 0.48 R the strategy must be right far more often than it is wrong just to
break even — a 45% win rate at 2R gross becomes a losing system. Three things
help, in order of effect: **trade the futures book rather than spot** (roughly
halves it), **widen the stop** (the hurdle scales inversely with stop
distance, and `stop_buffer_atr` is the lever), and **use a higher timeframe**
so the stop is naturally wider relative to price.

This is why the defaults are futures-shaped. A per-contract fee on ES is a few
percent of a typical stop; a percentage fee on BTC spot with a tight stop is
half of it.

### Risk sizing

`risk_per_trade` of 0.005 (0.5%) is the default and is already aggressive for a
strategy with no proven edge. Until you have **100+ out-of-sample trades with a
t-statistic above 2**, treat any live trading as paid research and size it at a
level where a 10-loss streak is irrelevant to you. The report prints the worst
losing streak for exactly this reason — six consecutive losses is normal in a
45%-win-rate system, and at 0.5% that is a 3% drawdown before anything has gone
wrong.

### Before going live

1. Backtest on **real tick data**, at least 6 months, ideally across a volatile
   and a quiet regime.
2. Confirm the run is not flagged `used_proxy_footprints`.
3. Run `optimize` and look at the **plateau table**, not the best row. If the
   good parameter values are isolated spikes, you have fitted noise.
4. Check the in-sample versus out-of-sample gap. A large gap means the search
   found nothing real.
5. Forward-test on live data with no money for a month. Compare the fills you
   would have gotten with the fills the backtest assumed. `paper` does this —
   see below.
6. Only then, size up slowly.

---

## `autopilot` — the set-and-forget strategy

The order-flow strategy is **not** a set-and-forget system. It needs tick data,
its cost hurdle eats roughly half its own risk on a percentage-fee venue, and it
would want revalidating every few months. It is a high-touch system wearing a
bot costume.

`autopilot` is the opposite, and the two are meant to coexist: this is the core
holding, the order-flow work is a satellite that has to earn its way in.

### The rule

Hold spot BTC while price closes above a long moving average. Hold nothing
otherwise. One number to choose, and you will not tune it.

```bash
# See what it would have done
python -m order_flow_strategy autopilot --binance-klines data/btc-daily --preset btc

# Run it forward on a daily schedule, book on disk, no orders placed
python -m order_flow_strategy autopilot --live \
    --binance-klines data/btc-daily --preset btc

# Look without touching
python -m order_flow_strategy autopilot --live --status-only \
    --binance-klines data/btc-daily --preset btc
```

### Three decisions, and why

**Binary exposure, not risk-based sizing.** A trend filter risking 0.5% per
signal and firing six times a year puts about 0.9% of the account at stake
annually. That is not an investment strategy, it is a rounding error with a cron
job. For a slow compounder the position *is* the account: fully in, or fully
out. That is also why compounding works here — each risk-on period puts the
grown equity to work, not a fixed fraction of it.

**Spot, not perp.** Funding at a baseline 0.01% per eight-hour settlement is
11.6% a year against a position held through a trend, and 0.05% is 72.8%.
Liquidation turns a survivable drawdown into a permanent loss. Neither risk buys
a set-and-forget core anything. Run a leveraged sleeve separately and
deliberately, if at all.

**Hysteresis, not a bare crossover.** Price must close a set fraction past the
average to flip the regime, so a market oscillating around the line does not
generate a trade per bar. Whipsaw, not trend accuracy, is what kills
moving-average systems. On generated data a 3% band cut regime flips from 18 to
10 over the same path.

### The safety rails, and the principle behind them

**A rail may block getting in. No rail may block getting out.** Being kept out
costs opportunity. Being kept in costs money — and the moments a rail is most
likely to misfire, a data gap or a violent bar or an outage, are exactly the
moments you most want the exit available.

| Rail | Blocks | Why |
|---|---|---|
| `warmup` | entry | The average is not populated yet |
| `min_hold` | entry | Refuses to re-buy within N bars of selling |
| `bad_data` | entry | A single bar moving more than `max_gap_pct` is suspect |
| `stale_data` | entry | The feed is behind; do not buy blind |
| `halted` | everything | A human must look before it resumes |

The drawdown halt is a **bug and bad-data alarm, not a stop loss.** Set it above
any drawdown the strategy legitimately produces. Set it too tight and it fires
at the bottom and locks you out of the recovery — which is exactly what happened
on the first generated run at a 45% threshold: the bot halted after 702 bars and
spent 65% of the sample locked flat while the report showed a headline return as
if it had traded throughout. The report now says so loudly, and the default is
looser. Backtest first, then set the number.

### Cost arithmetic

Binary exposure trades the whole account on every flip, so the drag is larger
than the same venue's drag on a risk-sized system:

```
annual drag = 2 × (fee + slippage) × flips_per_year × exposure
```

| Flips/year | At 15 bp per side |
|---|---|
| 4 | 1.2% |
| 6 | 1.8% |
| 12 | 3.6% |

This is the number that decides whether a slower average is worth using. A
longer period trades less and costs less, but enters and exits later.

### Operating it

- **Daily bars.** Faster timeframes multiply the flip count and the drag with it.
- **Pass the full history each run.** The runner folds in new bars incrementally,
  but the first run replays the history to work out which regime the market is
  currently in, then makes one trade to match. That is what deploying into an
  existing trend actually looks like.
- **Set-and-forget does not mean unmonitored.** It means no discretionary
  decisions. The failure mode of an unwatched bot is not a bad trade, it is
  silence while holding a position nobody is looking at. `status` prints a data
  age line — alarm on it.

### What is not yet known

Everything above is mechanism and arithmetic. On generated data the parameters
show a broad plateau from SMA 150 to 250 rather than an isolated spike, which is
the shape you want, but generated data has planted regimes and proves only that
the code works. **No run against real BTC bars exists yet.** Until one does,
treat the default of 200 as a reasonable prior, not a result.

---

### Forward-testing with `paper`

Forward tests cost calendar time, so start one before the backtest verdict is
in rather than after. It runs against fresh bars, keeps its book on disk, and
places nothing.

```bash
# Run it on a schedule. Once per bar close is enough.
python -m order_flow_strategy paper --binance-klines data/btc --preset btc \
    --state paper/state.json --journal paper/journal.jsonl

# Look at the book without touching it
python -m order_flow_strategy paper --binance-klines data/btc --preset btc \
    --status-only
```

It does not re-implement the trading rules. It replays the same engine over the
same bars, asks what position should be held *now*, and acts on the difference
against what it holds — reconciliation by construction, the same shape as
reconciling a live bot against an exchange. A second implementation could
silently disagree with the backtest and you would never know which was right.

Equity is read off the engine's curve for the same reason, and closed trades
are journalled with the engine's exit price and reason rather than a
close-price guess.

Three properties make it safe to schedule:

- **Restart safety.** State is on disk. A process that dies holding a position
  knows what it holds when it comes back. A corrupt state file raises rather
  than resetting — silently forgetting an open position is the worst available
  failure.
- **Idempotency.** Bars are processed once, keyed on timestamp. A retrying cron
  job cannot double a position.
- **An audit trail.** Every decision appends to a JSONL journal, including the
  ones that do nothing. Actions are `open_long`, `open_short`, `close`,
  `adjust`, `hold`, `no_signal`.

Pass the **full** bar history on every run, not a rolling window. The engine
restarts from `starting_equity` on each replay, so a shortened history rebases
the equity curve; the tool warns when the history it is handed is shorter than
the one before.

### When not to take the setup

- The level's evidence is only `wick_rejection` and `high_volume_node`. That is
  a chart pattern, not order flow — enforced by default, and worth respecting
  manually if you turn the gate off.
- Scheduled news inside the next few bars. Absorption means nothing when the
  passive seller is about to pull their order.
- The level has already been traded twice (`max_trades_per_level`). A level that
  needed three attempts is not holding.
- ATR has collapsed below half its median. Levels do not get tested properly in
  a dead market, and your stop is too tight for the eventual expansion.
- You are in a trend day and the level is counter-trend. This is the single most
  expensive mistake available in this strategy.

### Honest expectations

A rule set like this, working properly, looks like a **40–50% win rate with a
payoff ratio around 1.5–2.0**, which is a positive but modest expectancy that
costs and slippage can erase entirely. If your backtest shows a 70% win rate,
something is wrong with your data or your fills — go and find it.

---

## Configuration reference

Everything lives in `StrategyConfig` (`config.py`), which is frozen and copyable
via `cfg.with_(target_r=3.0)`. Distances are in ATR multiples wherever possible,
so a configuration transfers across instruments and volatility regimes. Run
`python -m order_flow_strategy backtest --help` for the CLI subset.

---

## Validating on your own data

```bash
python -m order_flow_strategy optimize --trades ticks.csv --folds 4
```

Anchored walk-forward: parameters are chosen on data that precedes the data they
are scored on. Three habits keep it honest:

- **Prefer plateaus to peaks.** The recommendation is the value with the best
  *median* score across all other settings, not the value in the single best run.
- **Report the degradation.** The in-sample and out-of-sample scores are printed
  side by side. A large gap is the finding.
- **Refuse to endorse.** If every configuration scores negative out of sample,
  the tool says "least-bad, NOT a recommendation" rather than dressing it up.

The ranking objective is `expectancy_R × sqrt(trades)`, proportional to a
t-statistic, so three lucky trades cannot win.

---

## Honest limitations

- **The synthetic tape proves the code runs, not that the strategy works.** The
  generator plants real passive blocks that price trades into, so absorption
  genuinely exists in the data — but there is no news, no other participants,
  and no reflexivity. Expectancy measured on it is meaningless.
- **Bar data cannot show absorption.** See [Data requirements](#data-requirements).
- **No queue modelling.** Limit targets are assumed to fill when touched. In
  reality you are at the back of the queue and the fills you miss are
  disproportionately the good ones. Real results will be worse than the backtest
  by more than the modelled slippage.
- **No overnight gaps, no session boundaries, no roll handling.** If your
  instrument gaps, the stop logic fills at the open, but nothing models the
  decision to hold through it.
- **Single position at a time**, no portfolio effects, no correlation.
- **Level detection is heuristic.** The weights are hand-set, not learned.

---

## Layout

| File | Purpose |
|---|---|
| `config.py` | Every tunable parameter, with validation. |
| `data.py` | Bars, trades, CSV loading, synthetic tape generator. |
| `footprint.py` | Volume at price by aggressor; delta, imbalances, clusters, POC. |
| `indicators.py` | Causal ATR, EMA, rolling quantiles, swing detection. |
| `levels.py` | Evidence detectors and the live level book. |
| `signals.py` | The two-candle pattern and its filters. |
| `backtest.py` | Bar-by-bar engine with pessimistic fills. |
| `metrics.py` | Performance statistics in R multiples. |
| `optimize.py` | Walk-forward search and recommendations. |
| `report.py` | Human-readable output. |
| `funding.py` | Perp funding settlements and their cash flows. |
| `baselines.py` | Trend-following and buy-and-hold benchmarks. |
| `confluence.py` | Which signal conditions actually earn their place. |
| `reporting_html.py` | Self-contained HTML report. |
| `paper.py` | Forward testing by reconciliation; state, journal, restart safety. |
| `autopilot.py` | Set-and-forget trend allocator: spot, binary exposure, safety rails. |
| `persistence.py` | Atomic state writes and append-only journals. |
| `presets.py` | Instrument conventions (`btc`, `es`). |
| `sources/binance.py` | Readers for Binance kline and aggTrades archives. |
| `sources/fetch.py` | Downloader for those archives. |
| `cli.py` | `backtest`, `optimize`, `scan`, `demo`, `compare`, `report`, `paper`, `autopilot`. |
| `tests/` | 288 unit tests, `unittest` only — no pytest required. |
