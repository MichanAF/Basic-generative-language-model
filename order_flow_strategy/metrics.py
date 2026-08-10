"""Performance statistics, reported in R multiples wherever possible.

R -- profit divided by the risk taken on that trade -- is the honest unit here,
because position size is a function of stop distance. Currency P&L conflates
"the rules worked" with "the stop happened to be tight".
"""

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, Sequence

from .backtest import BacktestResult, ClosedTrade
from .footprint import EPS


@dataclass
class PerformanceStats:
    n_trades: int = 0
    n_wins: int = 0
    n_losses: int = 0
    win_rate: float = 0.0
    expectancy_r: float = 0.0
    std_r: float = 0.0
    #: Expectancy divided by its own standard error. Roughly, how much of the
    #: result is signal rather than a lucky sample. Below ~2 means "unproven".
    t_stat: float = 0.0
    avg_win_r: float = 0.0
    avg_loss_r: float = 0.0
    payoff_ratio: float = 0.0
    profit_factor: float = 0.0
    total_pnl: float = 0.0
    return_pct: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0
    max_consecutive_losses: int = 0
    avg_bars_held: float = 0.0
    avg_mae_r_wins: float = 0.0
    avg_mae_r_losses: float = 0.0
    avg_mfe_r_losses: float = 0.0
    n_long: int = 0
    n_short: int = 0
    long_expectancy_r: float = 0.0
    short_expectancy_r: float = 0.0
    exit_reasons: Dict[str, int] = field(default_factory=dict)
    n_signals: int = 0
    signal_fill_rate: float = 0.0
    used_proxy_footprints: bool = False

    @property
    def is_statistically_thin(self) -> bool:
        return self.n_trades < 30 or abs(self.t_stat) < 2.0


def _mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _stdev(xs: Sequence[float]) -> float:
    if len(xs) < 2:
        return 0.0
    m = _mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def max_drawdown(curve: Sequence[float]) -> tuple:
    """``(absolute, fraction)`` peak-to-trough decline."""
    peak = float("-inf")
    worst_abs = 0.0
    worst_pct = 0.0
    for v in curve:
        peak = max(peak, v)
        dd = peak - v
        if dd > worst_abs:
            worst_abs = dd
        if peak > EPS:
            worst_pct = max(worst_pct, dd / peak)
    return worst_abs, worst_pct


def max_consecutive_losses(trades: Sequence[ClosedTrade]) -> int:
    worst = run = 0
    for t in trades:
        if t.pnl <= 0:
            run += 1
            worst = max(worst, run)
        else:
            run = 0
    return worst


def summarize(result: BacktestResult) -> PerformanceStats:
    trades = result.trades
    stats = PerformanceStats(
        n_signals=len(result.signals),
        used_proxy_footprints=result.used_proxy_footprints,
    )
    if result.signals:
        stats.signal_fill_rate = len(trades) / len(result.signals)
    if not trades:
        return stats

    rs = [t.r_multiple for t in trades]
    wins = [t for t in trades if t.pnl > 0]
    losses = [t for t in trades if t.pnl <= 0]

    stats.n_trades = len(trades)
    stats.n_wins = len(wins)
    stats.n_losses = len(losses)
    stats.win_rate = len(wins) / len(trades)
    stats.expectancy_r = _mean(rs)
    stats.std_r = _stdev(rs)
    if stats.std_r > EPS:
        stats.t_stat = stats.expectancy_r / (stats.std_r / math.sqrt(len(rs)))

    stats.avg_win_r = _mean([t.r_multiple for t in wins])
    stats.avg_loss_r = _mean([t.r_multiple for t in losses])
    if abs(stats.avg_loss_r) > EPS:
        stats.payoff_ratio = abs(stats.avg_win_r / stats.avg_loss_r)

    gross_win = sum(t.pnl for t in wins)
    gross_loss = abs(sum(t.pnl for t in losses))
    stats.profit_factor = gross_win / gross_loss if gross_loss > EPS else float("inf")

    stats.total_pnl = sum(t.pnl for t in trades)
    start = result.config.starting_equity
    stats.return_pct = stats.total_pnl / start if start > EPS else 0.0
    stats.max_drawdown, stats.max_drawdown_pct = max_drawdown(result.equity_curve)
    stats.max_consecutive_losses = max_consecutive_losses(trades)
    stats.avg_bars_held = _mean([t.bars_held for t in trades])

    stats.avg_mae_r_wins = _mean([t.mae_r for t in wins])
    stats.avg_mae_r_losses = _mean([t.mae_r for t in losses])
    stats.avg_mfe_r_losses = _mean([t.mfe_r for t in losses])

    longs = [t for t in trades if t.direction > 0]
    shorts = [t for t in trades if t.direction < 0]
    stats.n_long, stats.n_short = len(longs), len(shorts)
    stats.long_expectancy_r = _mean([t.r_multiple for t in longs])
    stats.short_expectancy_r = _mean([t.r_multiple for t in shorts])

    stats.exit_reasons = dict(Counter(t.exit_reason for t in trades))
    return stats


def robustness_score(stats: PerformanceStats, min_trades: int = 20) -> float:
    """Ranking objective used by the optimizer.

    Expectancy scaled by the square root of the sample size, which is
    proportional to the t-statistic. It rewards an edge that shows up
    repeatedly and refuses to reward three lucky trades.
    """
    if stats.n_trades < min_trades:
        return float("-inf")
    return stats.expectancy_r * math.sqrt(stats.n_trades)
