"""Human-readable output for backtests, live scans, and trade plans."""

from typing import List, Optional, Sequence

from .backtest import ClosedTrade
from .config import StrategyConfig
from .levels import Level
from .metrics import PerformanceStats
from .signals import SHORT, Signal

DISCLAIMER = (
    "Research tool, not financial advice. Backtested results are not a "
    "prediction; live fills, queue position, and data quality will all be worse "
    "than modelled here."
)

RULE = "=" * 72
THIN = "-" * 72


def _pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def format_stats(stats: PerformanceStats, title: str = "Backtest") -> str:
    if stats.n_trades == 0:
        return (
            f"{RULE}\n{title}\n{RULE}\n"
            f"No trades. {stats.n_signals} raw signals were generated"
            f"{' (all filtered out)' if stats.n_signals else ''}.\n"
            "Loosen the level or confirmation thresholds, or check that the data "
            "actually carries aggressor information.\n"
        )

    lines = [
        RULE,
        title,
        RULE,
        f"Trades              {stats.n_trades}  "
        f"({stats.n_long} long / {stats.n_short} short)",
        f"Win rate            {_pct(stats.win_rate)}  "
        f"({stats.n_wins}W / {stats.n_losses}L)",
        f"Expectancy          {stats.expectancy_r:+.3f} R per trade  "
        f"(sd {stats.std_r:.2f}, t {stats.t_stat:+.2f})",
        f"Avg win / avg loss  {stats.avg_win_r:+.2f} R / {stats.avg_loss_r:+.2f} R"
        f"  (payoff {stats.payoff_ratio:.2f})",
        f"Profit factor       {stats.profit_factor:.2f}",
        THIN,
        f"Net P&L             {stats.total_pnl:,.2f}  ({_pct(stats.return_pct)} of starting equity)",
        f"Max drawdown        {stats.max_drawdown:,.2f}  ({_pct(stats.max_drawdown_pct)})",
        f"Worst losing streak {stats.max_consecutive_losses}",
        f"Avg bars held       {stats.avg_bars_held:.1f}",
        THIN,
        f"Long expectancy     {stats.long_expectancy_r:+.3f} R",
        f"Short expectancy    {stats.short_expectancy_r:+.3f} R",
        f"Signals / fills     {stats.n_signals} / {stats.n_trades} "
        f"({_pct(stats.signal_fill_rate)} taken)",
        f"MAE on winners      {stats.avg_mae_r_wins:.2f} R "
        f"(how much heat the good trades took)",
        f"MFE on losers       {stats.avg_mfe_r_losses:.2f} R "
        f"(how far the bad ones ran first)",
        f"Exits               " + ", ".join(f"{k}: {v}" for k, v in sorted(stats.exit_reasons.items())),
    ]

    warnings = warnings_for(stats)
    if warnings:
        lines += [THIN, "Read this before believing any of the above:"]
        lines += [f"  ! {w}" for w in warnings]
    lines += [RULE, DISCLAIMER, ""]
    return "\n".join(lines)


def warnings_for(stats: PerformanceStats) -> List[str]:
    out: List[str] = []
    if stats.used_proxy_footprints:
        out.append(
            "Footprints were estimated from OHLCV, not measured from trades. "
            "Absorption cannot be detected this way, so the core premise of the "
            "strategy is not actually being tested."
        )
    if 0 < stats.n_trades < 30:
        out.append(f"Only {stats.n_trades} trades. Far too few to conclude anything.")
    if stats.n_trades >= 30 and abs(stats.t_stat) < 2.0:
        out.append(
            f"t-statistic {stats.t_stat:+.2f}: the sample is consistent with zero edge."
        )
    if stats.avg_mfe_r_losses >= 1.0:
        out.append(
            f"Losing trades averaged {stats.avg_mfe_r_losses:.2f} R in profit before failing. "
            "Consider scaling out earlier or tightening to breakeven sooner."
        )
    if stats.avg_mae_r_wins >= 0.8:
        out.append(
            f"Winners took {stats.avg_mae_r_wins:.2f} R of heat on average -- the stop is "
            "barely wide enough, and small slippage changes will flip outcomes."
        )
    if stats.exit_reasons.get("time_stop", 0) > 0.4 * max(stats.n_trades, 1):
        out.append(
            "Most trades ended on the time stop, meaning the reversal usually stalls "
            "rather than runs. A closer target may fit the behaviour better."
        )
    return out


def format_trades(trades: Sequence[ClosedTrade], limit: Optional[int] = 20) -> str:
    if not trades:
        return "No trades.\n"
    rows = list(trades)[: limit or len(trades)]
    lines = [
        RULE,
        f"Trades (showing {len(rows)} of {len(trades)})",
        RULE,
        f"{'#':>3} {'side':<5} {'entry':>10} {'exit':>10} {'qty':>6} "
        f"{'R':>7} {'P&L':>11} {'bars':>5}  reason",
    ]
    for i, t in enumerate(rows, 1):
        side = "short" if t.direction == SHORT else "long"
        lines.append(
            f"{i:>3} {side:<5} {t.entry_price:>10.2f} {t.exit_price:>10.2f} "
            f"{t.quantity:>6.0f} {t.r_multiple:>+7.2f} {t.pnl:>+11.2f} "
            f"{t.bars_held:>5}  {t.exit_reason}"
        )
    lines.append("")
    return "\n".join(lines)


def format_levels(levels: Sequence[Level], index: int, cfg: StrategyConfig, limit: int = 10) -> str:
    if not levels:
        return "No active levels.\n"
    lines = [RULE, "Active order-flow levels (strongest first)", RULE]
    for lv in levels[:limit]:
        eff = lv.effective_strength(index, cfg.level_half_life)
        kind = "RESISTANCE" if lv.side > 0 else "SUPPORT   "
        evidence = ", ".join(f"{k}" for k in sorted(lv.evidence))
        flip = "  (flipped)" if lv.flipped_from is not None else ""
        lines.append(
            f"{kind} {lv.price:>10.2f}  strength {eff:>5.1f}  "
            f"tests {lv.touches:>2} held {lv.holds:>2}{flip}\n"
            f"           evidence: {evidence}"
        )
    lines.append("")
    return "\n".join(lines)


def format_signal(sig: Signal, cfg: StrategyConfig) -> str:
    side = "SHORT" if sig.direction == SHORT else "LONG"
    stop_dist = abs(sig.stop - sig.entry_price)
    entry_desc = (
        "market on the close of candle 2"
        if sig.entry_mode == "close_of_2"
        else f"stop order at {sig.entry_price:.2f}, valid {cfg.entry_valid_bars} bars"
    )
    lines = [
        RULE,
        f"{side} setup confirmed on bar {sig.c2_index} (candle 1 was bar {sig.c1_index})",
        RULE,
        f"Entry     {sig.entry_price:>10.2f}   {entry_desc}",
        f"Stop      {sig.stop:>10.2f}   {stop_dist:.2f} away "
        f"({stop_dist / max(sig.atr, 1e-9):.2f} ATR)",
        f"Target    {sig.target:>10.2f}   {sig.r_multiple_target:.1f}R",
    ]
    if cfg.partial_at_r is not None:
        partial_price = (
            sig.entry_price - cfg.partial_at_r * sig.risk_per_unit
            if sig.direction == SHORT
            else sig.entry_price + cfg.partial_at_r * sig.risk_per_unit
        )
        lines.append(
            f"Scale     {partial_price:>10.2f}   take {cfg.partial_frac:.0%} at "
            f"{cfg.partial_at_r:.1f}R"
            + (", then stop to breakeven" if cfg.breakeven_after_partial else "")
        )
    lines.append("")
    lines.append("Why:")
    lines += [f"  - {r}" for r in sig.reasons]
    lines.append("")
    lines.append("Invalidation: a close back beyond the level, or the stop being touched.")
    lines.append("")
    return "\n".join(lines)
