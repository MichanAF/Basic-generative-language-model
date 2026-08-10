"""Bar-by-bar backtester with costs and deliberately pessimistic fill rules.

Design commitments, because these are what separate a backtest from a
flattering chart:

* **No lookahead.** Signals for bar ``i`` are computed from bars ``<= i`` only,
  and a position opened on the close of bar ``i`` is first managed on ``i + 1``.
* **Ambiguity resolves against you.** If a bar's range contains both the stop
  and the target, the stop is taken. Intrabar path is unknowable from OHLC, so
  the engine assumes the worse leg.
* **Gaps fill at the open**, not at the order price, in both directions.
* **Slippage on market and stop fills only.** Limit targets fill at the limit
  or better -- a resting limit does not slip against you, it simply may not
  fill, which the "stop wins ties" rule already accounts for.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

from .config import ENTRY_BREAK_OF_2, StrategyConfig
from .data import Bar
from .footprint import EPS
from .levels import Level
from .signals import SHORT, Indicators, Signal, SignalEngine, prepare


@dataclass
class ClosedTrade:
    direction: int
    entry_index: int
    entry_ts: float
    entry_price: float
    exit_index: int
    exit_ts: float
    exit_price: float
    quantity: float
    pnl: float
    r_multiple: float
    mae_r: float
    mfe_r: float
    bars_held: int
    exit_reason: str
    level_price: float
    level_strength: float
    reasons: List[str] = field(default_factory=list)

    @property
    def is_win(self) -> bool:
        return self.pnl > 0


@dataclass
class _Position:
    signal: Signal
    direction: int
    entry_index: int
    entry_price: float
    quantity: float
    initial_quantity: float
    risk_per_unit: float
    stop: float
    target: float
    level: Level
    partial_done: bool = False
    realized: float = 0.0
    best_price: float = 0.0
    worst_price: float = 0.0
    fills: int = 1

    def excursions(self) -> tuple:
        """``(mae_r, mfe_r)`` from the recorded price extremes."""
        risk = max(self.risk_per_unit, EPS)
        if self.direction == SHORT:
            mfe = (self.entry_price - self.best_price) / risk
            mae = (self.worst_price - self.entry_price) / risk
        else:
            mfe = (self.best_price - self.entry_price) / risk
            mae = (self.entry_price - self.worst_price) / risk
        return max(0.0, mae), max(0.0, mfe)


@dataclass
class _PendingEntry:
    signal: Signal
    placed_index: int
    expires_index: int


@dataclass
class BacktestResult:
    trades: List[ClosedTrade]
    equity_curve: List[float]
    bar_timestamps: List[float]
    signals: List[Signal]
    config: StrategyConfig
    bars_tested: int
    used_proxy_footprints: bool

    @property
    def final_equity(self) -> float:
        return self.equity_curve[-1] if self.equity_curve else self.config.starting_equity


class Backtester:
    def __init__(self, cfg: StrategyConfig):
        self.cfg = cfg
        self.point_value = cfg.tick_value / cfg.tick_size

    # ------------------------------------------------------------------
    def run(self, bars: Sequence[Bar], indicators: Optional[Indicators] = None) -> BacktestResult:
        cfg = self.cfg
        ind = indicators or prepare(bars, cfg)
        engine = SignalEngine(cfg)

        equity = cfg.starting_equity
        curve: List[float] = []
        trades: List[ClosedTrade] = []
        all_signals: List[Signal] = []
        position: Optional[_Position] = None
        pending: Optional[_PendingEntry] = None

        for i, bar in enumerate(bars):
            # 1. Manage a position opened on an earlier bar.
            if position is not None and position.entry_index < i:
                closed, equity = self._manage(position, i, bar, ind, equity)
                if closed is not None:
                    trades.append(closed)
                    self._apply_cooldown(position.level, closed, i)
                    position = None

            # 2. A resting stop-entry order from an earlier bar.
            if position is None and pending is not None and pending.placed_index < i:
                if i > pending.expires_index or self._pending_invalidated(pending, bar):
                    pending = None
                else:
                    position = self._try_trigger(pending, i, bar, equity)
                    if position is not None:
                        pending = None
                        # Entered intrabar, so the rest of this bar is still
                        # live. OHLC cannot say whether the bar's high came
                        # before or after the trigger, so a stop inside the
                        # bar counts as hit -- pessimistic by construction.
                        closed, equity = self._manage(
                            position, i, bar, ind, equity, same_bar_entry=True
                        )
                        if closed is not None:
                            trades.append(closed)
                            self._apply_cooldown(position.level, closed, i)
                            position = None

            # 3. Signals from this bar's close.
            for sig in engine.on_bar(i, bars, ind):
                all_signals.append(sig)
                if position is not None or pending is not None:
                    continue  # max_positions == 1
                if not engine.regime_allows(i, sig.direction, bars, ind):
                    continue
                if cfg.entry_mode == ENTRY_BREAK_OF_2:
                    pending = _PendingEntry(
                        signal=sig, placed_index=i, expires_index=i + cfg.entry_valid_bars
                    )
                else:
                    position = self._open(sig, i, sig.entry_price, equity)

            curve.append(self._mark_to_market(equity, position, bar))

        # Flatten anything still open at the end of the data.
        if position is not None:
            last_i = len(bars) - 1
            closed, equity = self._close(
                position, last_i, bars[last_i], bars[last_i].close, "end_of_data", equity
            )
            trades.append(closed)
            if curve:
                curve[-1] = equity

        return BacktestResult(
            trades=trades,
            equity_curve=curve,
            bar_timestamps=[b.ts for b in bars],
            signals=all_signals,
            config=cfg,
            bars_tested=len(bars),
            used_proxy_footprints=any(b.footprint.is_proxy for b in bars),
        )

    # ------------------------------------------------------------------
    # Entry
    # ------------------------------------------------------------------
    def _size(self, risk_per_unit: float, equity: float) -> float:
        cfg = self.cfg
        risk_cash = equity * cfg.risk_per_trade
        per_unit = max(risk_per_unit * self.point_value, EPS)
        qty = risk_cash / per_unit
        if cfg.whole_units:
            qty = float(int(qty))
        return max(qty, 0.0)

    def _open(self, sig: Signal, index: int, raw_price: float, equity: float) -> Optional[_Position]:
        cfg = self.cfg
        slip = cfg.slippage_ticks * cfg.tick_size
        fill = raw_price - slip if sig.direction == SHORT else raw_price + slip

        risk = (sig.stop - fill) if sig.direction == SHORT else (fill - sig.stop)
        if risk <= 0:
            return None
        qty = self._size(risk, equity)
        if qty <= 0:
            return None  # account too small to take this trade at this risk

        sig.level.trades_taken += 1
        return _Position(
            signal=sig,
            direction=sig.direction,
            entry_index=index,
            entry_price=fill,
            quantity=qty,
            initial_quantity=qty,
            risk_per_unit=risk,
            stop=sig.stop,
            target=(fill - cfg.target_r * risk) if sig.direction == SHORT
            else (fill + cfg.target_r * risk),
            level=sig.level,
            best_price=fill,
            worst_price=fill,
        )

    def _pending_invalidated(self, pending: _PendingEntry, bar: Bar) -> bool:
        """Kill a resting entry if price already invalidated the setup."""
        sig = pending.signal
        if sig.direction == SHORT:
            return bar.high >= sig.stop
        return bar.low <= sig.stop

    def _try_trigger(
        self, pending: _PendingEntry, index: int, bar: Bar, equity: float
    ) -> Optional[_Position]:
        sig = pending.signal
        trigger = sig.entry_price
        if sig.direction == SHORT:
            if bar.low > trigger:
                return None
            raw = min(bar.open, trigger)  # gap through the trigger fills at the open
        else:
            if bar.high < trigger:
                return None
            raw = max(bar.open, trigger)
        return self._open(sig, index, raw, equity)

    # ------------------------------------------------------------------
    # Management
    # ------------------------------------------------------------------
    def _manage(
        self,
        pos: _Position,
        index: int,
        bar: Bar,
        ind: Indicators,
        equity: float,
        same_bar_entry: bool = False,
    ):
        cfg = self.cfg

        if pos.direction == SHORT:
            pos.best_price = min(pos.best_price, bar.low)
            pos.worst_price = max(pos.worst_price, bar.high)
            stop_hit = bar.high >= pos.stop
            target_hit = bar.low <= pos.target
        else:
            pos.best_price = max(pos.best_price, bar.high)
            pos.worst_price = min(pos.worst_price, bar.low)
            stop_hit = bar.low <= pos.stop
            target_hit = bar.high >= pos.target

        # Stop first: when both are inside the bar, assume the bad one.
        if stop_hit:
            fill = self._stop_fill(pos, bar)
            reason = "stop" if not pos.partial_done else "stop_after_partial"
            if pos.partial_done and cfg.breakeven_after_partial:
                reason = "breakeven_after_partial"
            return self._close(pos, index, bar, fill, reason, equity)

        # Scale out.
        if cfg.partial_at_r is not None and not pos.partial_done:
            level = (
                pos.entry_price - cfg.partial_at_r * pos.risk_per_unit
                if pos.direction == SHORT
                else pos.entry_price + cfg.partial_at_r * pos.risk_per_unit
            )
            reached = bar.low <= level if pos.direction == SHORT else bar.high >= level
            if reached:
                fill = min(bar.open, level) if pos.direction == SHORT else max(bar.open, level)
                qty = pos.quantity * cfg.partial_frac
                equity += self._pnl(pos.direction, pos.entry_price, fill, qty)
                pos.quantity -= qty
                pos.realized += self._pnl(pos.direction, pos.entry_price, fill, qty)
                pos.partial_done = True
                pos.fills += 1
                if cfg.breakeven_after_partial:
                    pos.stop = pos.entry_price

        if target_hit:
            # Limit order: fills at the target, or better on a gap.
            fill = min(bar.open, pos.target) if pos.direction == SHORT else max(bar.open, pos.target)
            return self._close(pos, index, bar, fill, "target", equity)

        # Trail behind the favourable extreme.
        if cfg.trail_atr is not None:
            atr_value = max(ind.atr[index], EPS)
            if pos.direction == SHORT:
                pos.stop = min(pos.stop, bar.low + cfg.trail_atr * atr_value)
            else:
                pos.stop = max(pos.stop, bar.high - cfg.trail_atr * atr_value)

        # Time stop.
        if cfg.time_stop_bars is not None and not same_bar_entry:
            if index - pos.entry_index >= cfg.time_stop_bars:
                fill = self._market_fill(pos.direction, bar.close, exiting=True)
                return self._close(pos, index, bar, fill, "time_stop", equity)

        return None, equity

    def _stop_fill(self, pos: _Position, bar: Bar) -> float:
        """Stops slip, and a gap through one fills at the open."""
        cfg = self.cfg
        slip = cfg.slippage_ticks * cfg.tick_size
        if pos.direction == SHORT:
            return max(bar.open, pos.stop) + slip
        return min(bar.open, pos.stop) - slip

    def _market_fill(self, direction: int, price: float, exiting: bool) -> float:
        slip = self.cfg.slippage_ticks * self.cfg.tick_size
        if exiting:
            return price + slip if direction == SHORT else price - slip
        return price - slip if direction == SHORT else price + slip

    def _pnl(self, direction: int, entry: float, exit_price: float, qty: float) -> float:
        move = (entry - exit_price) if direction == SHORT else (exit_price - entry)
        gross = move * qty * self.point_value
        return gross - self.cfg.commission_per_side * qty * 2.0

    def _close(
        self, pos: _Position, index: int, bar: Bar, fill: float, reason: str, equity: float
    ):
        pnl_leg = self._pnl(pos.direction, pos.entry_price, fill, pos.quantity)
        equity += pnl_leg
        total_pnl = pos.realized + pnl_leg

        risk_cash = pos.risk_per_unit * pos.initial_quantity * self.point_value
        mae_r, mfe_r = pos.excursions()

        trade = ClosedTrade(
            direction=pos.direction,
            entry_index=pos.entry_index,
            entry_ts=pos.signal.ts,
            entry_price=pos.entry_price,
            exit_index=index,
            exit_ts=bar.ts,
            exit_price=fill,
            quantity=pos.initial_quantity,
            pnl=total_pnl,
            r_multiple=total_pnl / risk_cash if risk_cash > EPS else 0.0,
            mae_r=mae_r,
            mfe_r=mfe_r,
            bars_held=index - pos.entry_index,
            exit_reason=reason,
            level_price=pos.level.price,
            level_strength=pos.level.strength,
            reasons=list(pos.signal.reasons),
        )
        return trade, equity

    def _apply_cooldown(self, level: Level, trade: ClosedTrade, index: int) -> None:
        if trade.pnl <= 0:
            level.cooldown_until = index + self.cfg.cooldown_bars

    def _mark_to_market(self, equity: float, pos: Optional[_Position], bar: Bar) -> float:
        if pos is None:
            return equity
        move = (
            (pos.entry_price - bar.close) if pos.direction == SHORT
            else (bar.close - pos.entry_price)
        )
        return equity + move * pos.quantity * self.point_value
