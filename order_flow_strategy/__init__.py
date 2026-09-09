"""Order-flow resistance / two-candle reversal trading strategy.

A self-contained research toolkit that:

1. Reads order flow (tick trades with an aggressor side, or OHLCV bars via a
   documented proxy) and builds per-bar footprints.
2. Locates where supply and demand actually sit -- absorption, stacked
   imbalances, high volume nodes, delta divergence -- instead of drawing
   horizontal lines by eye.
3. Waits for a two-candle reversal at one of those levels: candle 1 tests and
   is rejected, candle 2 confirms and is where the position is placed.
4. Backtests the rules bar by bar with costs and no lookahead, then reports
   performance and parameter recommendations.

Nothing here is financial advice. See ``README.md`` in this directory for the
full specification, the data requirements, and the honest limitations.
"""

from .config import StrategyConfig
from .data import Bar, Trade, bars_from_trades, load_bars_csv, load_trades_csv
from .footprint import Footprint
from .levels import Level, LevelBook
from .signals import Setup, Signal, SignalEngine
from .backtest import Backtester, BacktestResult, ClosedTrade, OpenPosition
from .metrics import PerformanceStats, summarize
from .paper import Decision, PaperState, PaperTrader

__all__ = [
    "StrategyConfig",
    "Bar",
    "Trade",
    "Footprint",
    "Level",
    "LevelBook",
    "Setup",
    "Signal",
    "SignalEngine",
    "Backtester",
    "BacktestResult",
    "ClosedTrade",
    "OpenPosition",
    "PerformanceStats",
    "summarize",
    "PaperTrader",
    "PaperState",
    "Decision",
    "bars_from_trades",
    "load_bars_csv",
    "load_trades_csv",
]

__version__ = "1.0.0"
