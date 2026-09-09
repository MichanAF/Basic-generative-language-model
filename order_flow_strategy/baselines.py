"""Benchmarks. Without these, a strategy result means nothing.

The order-flow strategy is complicated. Complicated things feel like they must
be earning their keep, and the only way to find out is to put something plain
next to them. Two plain things live here:

``buy_and_hold``
    One trade, held throughout. On BTC this is a brutally hard benchmark to
    beat on raw return, and pretending otherwise is how people justify systems
    that underperform doing nothing.

``trend_baseline``
    Long while price is above its moving average, flat otherwise. Two
    parameters, one of which you will not tune. This is the strategy the
    evidence actually supports on a trending asset, and it is the honest
    yardstick for anything more elaborate.

Both emit a :class:`BacktestResult`, so ``summarize`` and ``format_stats``
report them in exactly the same terms as the order-flow engine -- same cost
model, same R multiples, same warnings.
"""

from typing import List, Optional, Sequence

from .backtest import BacktestResult, Backtester, ClosedTrade, OpenPosition
from .config import StrategyConfig
from .data import Bar
from .footprint import EPS
from .indicators import atr as compute_atr
from .funding import FundingSchedule
from .signals import LONG

#: Size by stop distance, so R multiples compare with the order-flow engine.
SIZE_RISK = "risk"
#: Commit all equity, which is what a spot core position actually does.
SIZE_FULL = "full"


def _sma(values: Sequence[float], period: int) -> List[Optional[float]]:
    """Causal simple moving average; None until the window fills."""
    out: List[Optional[float]] = []
    running = 0.0
    for i, v in enumerate(values):
        running += v
        if i >= period:
            running -= values[i - period]
        out.append(running / period if i >= period - 1 else None)
    return out


def _make_trade(
    bt: Backtester,
    entry_index: int,
    entry_bar: Bar,
    entry_price: float,
    exit_index: int,
    exit_bar: Bar,
    exit_price: float,
    qty: float,
    risk_per_unit: float,
    reason: str,
    best: float,
    worst: float,
    funding_cash: float,
    reasons: List[str],
) -> ClosedTrade:
    gross = bt._gross(LONG, entry_price, exit_price, qty)
    cost = (
        bt._commission(entry_price, qty)
        + bt._commission(exit_price, qty)
        + 2 * bt._slippage(entry_price) * qty * bt.point_value
    )
    pnl = gross - cost + funding_cash
    risk_cash = max(risk_per_unit * qty * bt.point_value, EPS)
    return ClosedTrade(
        direction=LONG,
        entry_index=entry_index,
        entry_ts=entry_bar.ts,
        entry_price=entry_price,
        exit_index=exit_index,
        exit_ts=exit_bar.ts,
        exit_price=exit_price,
        quantity=qty,
        pnl=pnl,
        r_multiple=pnl / risk_cash,
        mae_r=max(0.0, (entry_price - worst) / max(risk_per_unit, EPS)),
        mfe_r=max(0.0, (best - entry_price) / max(risk_per_unit, EPS)),
        bars_held=exit_index - entry_index,
        exit_reason=reason,
        level_price=entry_price,
        level_strength=0.0,
        cost=cost,
        cost_r=cost / risk_cash,
        funding=funding_cash,
        reasons=reasons,
    )


def trend_baseline(
    bars: Sequence[Bar],
    cfg: StrategyConfig,
    sma_period: int = 200,
    size_mode: str = SIZE_RISK,
    funding: Optional[FundingSchedule] = None,
    flatten_at_end: bool = True,
) -> BacktestResult:
    """Long while close > SMA, flat otherwise. Long-only, no shorts.

    The moving average is both the entry trigger and the exit, so the distance
    from entry to the average is the natural risk on the trade -- that is what
    sizes the position and defines R.

    On BTC the long-only restriction is deliberate: the return distribution is
    dominated by a handful of violent up-moves, and systematically shorting
    into them has historically been where crypto systematic traders lose.
    """
    if size_mode not in (SIZE_RISK, SIZE_FULL):
        raise ValueError(f"size_mode must be {SIZE_RISK!r} or {SIZE_FULL!r}")
    if sma_period < 2:
        raise ValueError("sma_period must be at least 2")

    bt = Backtester(cfg)
    bt.funding = funding
    sma = _sma([b.close for b in bars], sma_period)
    # A cross happens *at* the average, so entry-to-average can be arbitrarily
    # small. Left unfloored that produces near-zero risk and R multiples in the
    # hundreds -- arithmetic, not edge. Floor the stop a full ATR away, which is
    # also what you would actually place.
    atr_vals = compute_atr(bars, cfg.atr_period)

    equity = cfg.starting_equity
    curve: List[float] = []
    trades: List[ClosedTrade] = []

    open_i: Optional[int] = None
    entry_price = qty = risk_per_unit = 0.0
    best = worst = 0.0
    funding_cash = 0.0
    last_funding_ts = 0.0

    for i, bar in enumerate(bars):
        avg = sma[i]

        # Funding accrues on whatever is open, before any exit decision.
        if open_i is not None and funding is not None and not funding.is_empty:
            flow = funding.cash_flow(
                LONG, qty * bar.close * bt.point_value, last_funding_ts, bar.ts
            )
            funding_cash += flow
            equity += flow
        if open_i is not None:
            last_funding_ts = bar.ts
            best = max(best, bar.high)
            worst = min(worst, bar.low)

        if avg is not None:
            if open_i is None and bar.close > avg:
                # Enter on the close that crosses up; the average is the stop.
                raw = bar.close
                fill = raw + bt._slippage(raw)
                risk_per_unit = max(fill - avg, atr_vals[i], EPS)
                if size_mode == SIZE_RISK:
                    qty = bt._size(risk_per_unit, equity, price=fill)
                else:
                    qty = equity / (fill * bt.point_value)
                    if cfg.whole_units:
                        qty = float(int(qty))
                if qty > 0:
                    open_i, entry_price = i, fill
                    best = worst = fill
                    funding_cash = 0.0
                    last_funding_ts = bar.ts

            elif open_i is not None and bar.close < avg:
                raw = bar.close
                fill = raw - bt._slippage(raw)
                trade = _make_trade(
                    bt, open_i, bars[open_i], entry_price, i, bar, fill, qty,
                    risk_per_unit, "trend_exit", best, worst, funding_cash,
                    [f"close {bar.close:.2f} below SMA{sma_period} {avg:.2f}"],
                )
                equity += trade.pnl - funding_cash  # funding already applied
                trades.append(trade)
                open_i = None

        mark = equity
        if open_i is not None:
            mark += (bar.close - entry_price) * qty * bt.point_value
        curve.append(mark)

    still_open: Optional[OpenPosition] = None
    if open_i is not None and not flatten_at_end:
        still_open = OpenPosition(
            direction=LONG, entry_index=open_i, entry_ts=bars[open_i].ts,
            entry_price=entry_price, quantity=qty,
            stop=entry_price - risk_per_unit, target=0.0,
        )
    elif open_i is not None:
        last = len(bars) - 1
        bar = bars[last]
        fill = bar.close - bt._slippage(bar.close)
        trade = _make_trade(
            bt, open_i, bars[open_i], entry_price, last, bar, fill, qty,
            risk_per_unit, "end_of_data", best, worst, funding_cash,
            ["still long when the data ran out"],
        )
        equity += trade.pnl - funding_cash
        trades.append(trade)
        if curve:
            curve[-1] = equity

    return BacktestResult(
        trades=trades,
        equity_curve=curve,
        bar_timestamps=[b.ts for b in bars],
        signals=[],
        config=cfg,
        bars_tested=len(bars),
        used_proxy_footprints=any(b.footprint.is_proxy for b in bars),
        open_position=still_open,
    )


def buy_and_hold(bars: Sequence[Bar], cfg: StrategyConfig) -> BacktestResult:
    """Buy the first bar, sell the last. The benchmark that must be beaten.

    Risk is defined as the full position value, so an R multiple here reads as
    "fraction of the stake made or lost" -- there is no stop to measure against.
    """
    if len(bars) < 2:
        raise ValueError("need at least two bars")

    bt = Backtester(cfg)
    first, last = bars[0], bars[-1]
    entry = first.close + bt._slippage(first.close)
    exit_price = last.close - bt._slippage(last.close)

    qty = cfg.starting_equity / (entry * bt.point_value)
    if cfg.whole_units:
        qty = max(1.0, float(int(qty)))

    trade = _make_trade(
        bt, 0, first, entry, len(bars) - 1, last, exit_price, qty,
        entry,  # risk == full position value
        "end_of_data",
        max(b.high for b in bars), min(b.low for b in bars), 0.0,
        ["bought the first bar, sold the last"],
    )

    curve = [
        cfg.starting_equity + (b.close - entry) * qty * bt.point_value for b in bars
    ]
    curve[-1] = cfg.starting_equity + trade.pnl

    return BacktestResult(
        trades=[trade],
        equity_curve=curve,
        bar_timestamps=[b.ts for b in bars],
        signals=[],
        config=cfg,
        bars_tested=len(bars),
        used_proxy_footprints=any(b.footprint.is_proxy for b in bars),
    )
