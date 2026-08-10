"""Every tunable knob for the strategy, in one place.

The defaults are a starting point for a liquid futures contract on a 5 minute
chart, not a tuned edge. Run ``python -m order_flow_strategy optimize`` on your
own data before trusting any of these numbers.
"""

from dataclasses import dataclass, replace
from typing import Optional

# Entry timing modes.
ENTRY_CLOSE_OF_2 = "close_of_2"
ENTRY_BREAK_OF_2 = "break_of_2"
ENTRY_MODES = (ENTRY_CLOSE_OF_2, ENTRY_BREAK_OF_2)

# Regime filters.
REGIME_NONE = "none"
REGIME_COUNTER_TREND_ONLY = "counter_trend_only"
REGIME_WITH_TREND_ONLY = "with_trend_only"
REGIME_FILTERS = (REGIME_NONE, REGIME_COUNTER_TREND_ONLY, REGIME_WITH_TREND_ONLY)


@dataclass(frozen=True)
class StrategyConfig:
    """Immutable strategy parameters.

    Distances are expressed in ATR multiples wherever possible so the same
    configuration transfers across instruments and volatility regimes.
    """

    # ------------------------------------------------------------------
    # Instrument
    # ------------------------------------------------------------------
    tick_size: float = 0.25
    #: Currency value of one tick for one unit/contract.
    tick_value: float = 12.50
    atr_period: int = 14

    # ------------------------------------------------------------------
    # Level discovery -- "where does resistance sit?"
    # ------------------------------------------------------------------
    #: Bars of history used for the rolling volume profile and percentiles.
    profile_lookback: int = 240
    #: A price level is a high volume node if its profile volume ranks at or
    #: above this quantile of the lookback profile.
    hvn_percentile: float = 0.85
    #: Bar volume must rank at or above this quantile to qualify as absorption.
    absorption_volume_pct: float = 0.75
    #: For resistance, the bar must close in the bottom fraction of its range.
    absorption_close_frac: float = 0.35
    #: Aggressive buy volume must be at least this multiple of aggressive sell
    #: volume inside the tested cluster for the print to read as absorption
    #: (buyers were aggressive, price still failed).
    absorption_flow_ratio: float = 1.15
    #: Diagonal footprint imbalance ratio (ask[p] vs bid[p-1 tick]).
    imbalance_ratio: float = 3.0
    #: Minimum volume on both sides of an imbalance comparison to count.
    imbalance_min_volume: float = 2.0
    #: Consecutive imbalanced price levels needed for a "stack".
    imbalance_stack: int = 3
    #: Fraction of the bar range treated as the tested cluster (top/bottom).
    cluster_frac: float = 0.30
    #: Two levels within this ATR distance are merged into one.
    level_merge_atr: float = 0.25
    #: Bars over which a level's strength halves if it is never retested.
    level_half_life: int = 120
    #: Levels below this strength are dropped and never traded.
    level_min_strength: float = 1.5
    #: Only trade levels carrying at least one genuine order-flow signature
    #: (absorption, stacked imbalance, or delta divergence). Wicks, volume
    #: nodes and retests alone make a chart pattern, not an order-flow level.
    #: Turning this off is a useful experiment and a bad default.
    require_flow_backed_levels: bool = True
    #: A close beyond a level by this many ATR invalidates (and may flip) it.
    level_break_atr: float = 0.50
    #: Cap on how many levels per side are carried forward.
    max_levels_per_side: int = 8
    #: Swing lookback used for the delta divergence test.
    swing_lookback: int = 5

    # ------------------------------------------------------------------
    # Candle 1 -- the test and rejection
    # ------------------------------------------------------------------
    #: How close the bar extreme must come to the level, in ATR.
    touch_tol_atr: float = 0.20
    #: Rejection wick must be at least this fraction of the bar range.
    wick_frac: float = 0.35
    #: Normalized delta (delta / volume) ceiling for a resistance test.
    c1_delta_max: float = 0.10
    #: Require at least one order flow confirmation, not just the wick.
    require_flow_evidence: bool = True
    #: Bars an armed setup stays valid waiting for candle 2.
    setup_expiry_bars: int = 1

    # ------------------------------------------------------------------
    # Candle 2 -- the confirmation, and where the trade is placed
    # ------------------------------------------------------------------
    #: Candle 2 body must be at least this fraction of its range.
    c2_body_frac_min: float = 0.30
    #: Candle 2 must close below this fraction of candle 1's range (measured
    #: from candle 1's low) for a short. Mirrored for longs.
    c2_close_beyond_c1_frac: float = 0.50
    #: Normalized delta ceiling for candle 2 on a short.
    c2_delta_max: float = 0.0
    #: Require candle 2's delta to be weaker than candle 1's (sellers pressing).
    c2_delta_must_worsen: bool = False
    #: If False, candle 2 taking out candle 1's extreme kills the setup.
    allow_c2_new_extreme: bool = False
    entry_mode: str = ENTRY_CLOSE_OF_2
    #: For ``break_of_2``: bars the resting stop order stays live.
    entry_valid_bars: int = 2

    # ------------------------------------------------------------------
    # Risk and trade management
    # ------------------------------------------------------------------
    #: Stop sits beyond max(level, candle 1 extreme) by this many ATR.
    stop_buffer_atr: float = 0.25
    #: Final target as a multiple of initial risk (R).
    target_r: float = 2.0
    #: Scale out this fraction at ``partial_at_r``. None disables scaling out.
    partial_at_r: Optional[float] = 1.0
    partial_frac: float = 0.5
    #: Move the stop to entry once the partial is taken.
    breakeven_after_partial: bool = True
    #: Trail the stop this many ATR behind the extreme. None disables trailing.
    trail_atr: Optional[float] = None
    #: Flatten after this many bars in the trade. None disables the time stop.
    time_stop_bars: Optional[int] = 20
    #: Fraction of equity risked per trade.
    risk_per_trade: float = 0.005
    starting_equity: float = 100_000.0
    #: Round position size down to whole units (futures contracts, shares).
    whole_units: bool = True
    max_positions: int = 1
    #: Bars to wait after a loss before re-arming the same level.
    cooldown_bars: int = 5
    #: Cap on how many times one level may be traded.
    max_trades_per_level: int = 2

    # ------------------------------------------------------------------
    # Filters
    # ------------------------------------------------------------------
    regime_filter: str = REGIME_NONE
    ema_period: int = 50
    #: Skip setups when ATR is below this fraction of its own median.
    min_atr_frac: float = 0.50
    #: Optional session window, minutes from midnight UTC. None disables.
    session_start_min: Optional[int] = None
    session_end_min: Optional[int] = None
    #: Trade only shorts at resistance, only longs at support, or both.
    trade_shorts: bool = True
    trade_longs: bool = True

    # ------------------------------------------------------------------
    # Costs
    #
    # Both a fixed and a proportional component, because the two market types
    # this targets charge differently. Futures are per contract, so the fixed
    # terms fit. Crypto venues charge basis points of notional, and over a
    # multi-year sample the underlying can move several-fold -- a fixed
    # per-unit fee would misstate costs by that same factor. Set whichever
    # matches your venue; they add, and leaving one at zero disables it.
    # ------------------------------------------------------------------
    #: Slippage against the trade, in ticks, per market/stop fill.
    slippage_ticks: float = 1.0
    #: Additional slippage as a fraction of price (0.0001 = 1 bp).
    slippage_pct: float = 0.0
    #: Commission in currency per unit per side.
    commission_per_side: float = 2.50
    #: Additional commission as a fraction of notional per side
    #: (0.0005 = 5 bp, roughly a crypto taker fee).
    commission_pct: float = 0.0

    def __post_init__(self) -> None:
        if self.entry_mode not in ENTRY_MODES:
            raise ValueError(f"entry_mode must be one of {ENTRY_MODES}")
        if self.regime_filter not in REGIME_FILTERS:
            raise ValueError(f"regime_filter must be one of {REGIME_FILTERS}")
        if self.tick_size <= 0:
            raise ValueError("tick_size must be positive")
        if not 0 < self.risk_per_trade < 1:
            raise ValueError("risk_per_trade must be in (0, 1)")
        if self.partial_at_r is not None:
            if not 0 < self.partial_frac < 1:
                raise ValueError("partial_frac must be in (0, 1)")
            if self.partial_at_r >= self.target_r:
                raise ValueError("partial_at_r must be below target_r")
        if not 0 < self.cluster_frac <= 0.5:
            raise ValueError("cluster_frac must be in (0, 0.5]")
        if self.slippage_pct < 0 or self.commission_pct < 0:
            raise ValueError("percentage costs cannot be negative")
        if not self.trade_shorts and not self.trade_longs:
            raise ValueError("at least one of trade_shorts / trade_longs must be on")

    def with_(self, **overrides) -> "StrategyConfig":
        """Return a copy with ``overrides`` applied."""
        return replace(self, **overrides)
