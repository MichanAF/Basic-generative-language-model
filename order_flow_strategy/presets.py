"""Instrument presets.

The defaults in :class:`StrategyConfig` describe a liquid futures contract. A
crypto pair differs in four ways that all matter, and getting any of them wrong
quietly invalidates a backtest:

1. **Row size.** ``tick_size`` is the footprint row height, not the exchange's
   price increment. BTCUSDT ticks at $0.01; at that row size a five minute bar
   spanning $300 has 30,000 rows, which is neither readable nor computable.
   $10 rows give roughly 20-40 rows per bar, which is what a real footprint
   chart shows.
2. **Fractional size.** You can hold 0.37 BTC. You cannot hold 0.37 ES
   contracts, so ``whole_units`` flips off.
3. **Proportional costs.** Crypto venues charge basis points of notional. Over
   two years BTC ranged roughly $16k to $120k, so a fixed per-unit fee would
   misstate costs by that factor across the sample.
4. **No session.** Crypto trades continuously, so there is no session window to
   filter and no overnight gap. Level half-life is lengthened accordingly --
   120 bars is about six hours on a 5m chart, which is a session on a futures
   contract but a fraction of a day here.
"""

from typing import Dict

from .config import StrategyConfig

#: Binance spot taker fee at the base VIP tier, per side.
BINANCE_TAKER_FEE = 0.001


def btc_5m(**overrides) -> StrategyConfig:
    """BTCUSDT on 5 minute bars, Binance spot conventions.

    ``tick_value`` equals ``tick_size`` so one unit is one BTC and a $1 move on
    one unit is $1 of P&L.
    """
    base = dict(
        tick_size=10.0,          # footprint row height in dollars
        tick_value=10.0,         # => point_value 1.0: one unit is one BTC
        whole_units=False,
        # Spot cannot borrow. Raise this deliberately if you are on perps and
        # have decided you want leverage -- do not inherit it by accident.
        max_leverage=1.0,
        commission_per_side=0.0,
        commission_pct=BINANCE_TAKER_FEE,
        slippage_ticks=0.0,
        slippage_pct=0.0002,     # 2 bp, generous for BTCUSDT top of book
        session_start_min=None,
        session_end_min=None,
        level_half_life=288,     # 24 hours of 5m bars
        profile_lookback=288,
        time_stop_bars=24,       # two hours
    )
    base.update(overrides)
    return StrategyConfig(**base)


def es_5m(**overrides) -> StrategyConfig:
    """E-mini S&P 500 futures on 5 minute bars, RTH only."""
    base = dict(
        tick_size=0.25,
        tick_value=12.50,
        whole_units=True,
        commission_per_side=2.50,
        commission_pct=0.0,
        slippage_ticks=1.0,
        session_start_min=810,   # 13:30 UTC
        session_end_min=1200,    # 20:00 UTC
    )
    base.update(overrides)
    return StrategyConfig(**base)


PRESETS: Dict[str, callable] = {"btc": btc_5m, "es": es_5m}


def build(name: str, **overrides) -> StrategyConfig:
    if name not in PRESETS:
        raise ValueError(f"unknown preset {name!r}; choose from {sorted(PRESETS)}")
    return PRESETS[name](**overrides)
