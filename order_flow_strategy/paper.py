"""Paper trading: run the rules forward against live data, place nothing.

A backtest tells you what the rules did to history. It cannot tell you whether
your fills resemble the fills it assumed, whether the data arrives when you
expect, or whether the thing survives being restarted at 3am. Those only show
up going forward, and forward costs calendar time -- which is why this is worth
starting before the backtest verdict is in, not after.

**The design decision that matters.** This does not re-implement the trading
logic. Re-implementing it would create a second version that can silently
disagree with the backtest, and you would never know which one was right.
Instead it replays the same engine over the same bars, asks it what position
should be held *now*, and acts on the difference against what it currently
holds. Reconciliation by construction -- the same shape as reconciling a live
bot against an exchange.

The same reasoning governs money. Equity is read off the engine's own curve
rather than recomputed here. A paper book that marks its own exits at the
close would disagree with a backtest that exits at the stop mid-bar, and the
disagreement would be an artefact of the second implementation, not a finding.
Realised trades are journalled with the engine's exit price and reason.

**What it guarantees.**

*Restart safety.* All state is on disk and read back on start. A process that
dies mid-position knows what it holds when it comes back.

*Idempotency.* Bars are processed once, keyed on timestamp. Running twice on
the same data produces no second decision -- the property that stops a
retrying cron job from doubling a position.

*An audit trail.* Every decision, including "do nothing", appends to a
journal. The journal is the evidence; a paper trader you cannot audit
afterwards has told you nothing.

**What it requires of the caller.** Pass the full bar history every time, not
a rolling window. The engine restarts from ``starting_equity`` on each replay,
so a shortened history silently rebases the equity curve. ``update`` warns
when the history it is handed is shorter than the one before.

Nothing here touches an exchange or the network.
"""

import json
import time
from dataclasses import asdict, dataclass
from typing import Callable, List, Optional, Sequence

from .backtest import BacktestResult, Backtester, OpenPosition
from .config import StrategyConfig
from .data import Bar
from .footprint import EPS
from .funding import FundingSchedule
from .persistence import append_jsonl, atomic_write_json, load_dataclass, read_jsonl

#: A runner takes bars and returns a result carrying ``open_position``.
Runner = Callable[[Sequence[Bar]], BacktestResult]

FLAT = 0

#: Below this, a stop or target move is rounding noise, not an amendment
#: worth sending to an exchange.
AMEND_EPS = 1e-9


@dataclass
class Decision:
    """One thing the bot would have done, or explicitly chose not to do."""

    bar_ts: float
    decided_at: float
    #: open_long | open_short | close | adjust | hold | no_signal
    action: str
    direction: int = 0
    quantity: float = 0.0
    price: float = 0.0
    stop: float = 0.0
    target: float = 0.0
    equity: float = 0.0
    #: Realised R, on ``close`` decisions only.
    r_multiple: float = 0.0
    reason: str = ""

    def line(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)


@dataclass
class PaperState:
    """Everything needed to resume after a restart."""

    equity: float = 0.0
    direction: int = FLAT
    quantity: float = 0.0
    entry_price: float = 0.0
    stop: float = 0.0
    target: float = 0.0
    #: Entry timestamp doubles as the position's identity. Two longs opened at
    #: different times are different positions, and treating them as one hides
    #: a whole round trip.
    entry_ts: float = 0.0
    #: Newest bar already processed. The idempotency key.
    last_bar_ts: float = 0.0
    #: Trades the engine had already closed at the last update, so that only
    #: genuinely new closures get journalled.
    n_trades_seen: int = 0
    bars_seen: int = 0
    n_decisions: int = 0
    started_at: float = 0.0

    @property
    def is_flat(self) -> bool:
        return self.direction == FLAT or self.quantity <= EPS

    def unrealised(self, price: float, point_value: float) -> float:
        if self.is_flat:
            return 0.0
        move = (
            (price - self.entry_price)
            if self.direction > 0
            else (self.entry_price - price)
        )
        return move * self.quantity * point_value


def load_state(path: str, starting_equity: float) -> PaperState:
    """Read state from disk, or start fresh. A corrupt file is fatal.

    Silently resetting on a parse error would mean a bot that quietly forgets
    an open position -- the worst possible failure for something holding risk.
    """
    loaded = load_dataclass(path, PaperState, what="position")
    if loaded is None:
        return PaperState(equity=starting_equity, started_at=time.time())
    return loaded


def save_state(path: str, state: PaperState) -> None:
    """Write atomically, so a crash mid-write cannot corrupt the state."""
    atomic_write_json(path, state)


class PaperTrader:
    """Forward-tests a strategy without placing an order."""

    def __init__(
        self,
        cfg: StrategyConfig,
        state_path: str = "paper/state.json",
        journal_path: str = "paper/journal.jsonl",
        runner: Optional[Runner] = None,
        funding: Optional[FundingSchedule] = None,
    ):
        self.cfg = cfg
        self.state_path = state_path
        self.journal_path = journal_path
        self.point_value = cfg.tick_value / cfg.tick_size
        self.funding = funding
        self.runner = runner or self._default_runner
        self.state = load_state(state_path, cfg.starting_equity)
        #: Non-fatal problems from the last update, for the caller to print.
        self.warnings: List[str] = []

    def _default_runner(self, bars: Sequence[Bar]) -> BacktestResult:
        return Backtester(self.cfg).run(
            bars, funding=self.funding, flatten_at_end=False
        )

    # ------------------------------------------------------------------
    def update(self, bars: Sequence[Bar]) -> List[Decision]:
        """Process any new bars and return the decisions taken.

        Returns an empty list when there is nothing new -- the normal outcome,
        and the one that makes it safe to run this on a tight schedule.
        """
        self.warnings = []
        if not bars:
            return []
        newest = bars[-1].ts
        if newest <= self.state.last_bar_ts:
            return []  # already seen; do not act twice on one bar

        if self.state.bars_seen and len(bars) < self.state.bars_seen:
            self.warnings.append(
                f"history shrank from {self.state.bars_seen:,} bars to "
                f"{len(bars):,}. The engine restarts from starting_equity on "
                "every replay, so equity is now rebased and not comparable "
                "with earlier journal entries."
            )

        result = self.runner(bars)
        decisions = self._closures(result, bars[-1])
        if result.equity_curve:
            self.state.equity = result.equity_curve[-1]
        decisions += self._reconcile(result.open_position, bars[-1])

        self.state.last_bar_ts = newest
        self.state.bars_seen = len(bars)
        self.state.n_trades_seen = len(result.trades)
        self.state.n_decisions += len(decisions)
        for d in decisions:
            self._journal(d)
        save_state(self.state_path, self.state)
        return decisions

    # ------------------------------------------------------------------
    def _closures(self, result: BacktestResult, bar: Bar) -> List[Decision]:
        """Journal trades the engine closed since the last update.

        The exit price and reason come from the engine, so the journal records
        what the rules actually did rather than this module's guess at it.

        The first update is a bootstrap: every trade in the replayed history is
        "new" to an empty state file, but those are backtest results, not
        decisions this bot took forward. They are counted, not journalled.
        """
        fresh = result.trades[self.state.n_trades_seen :]
        if self.state.bars_seen == 0:
            if fresh:
                self.warnings.append(
                    f"first run: adopted {len(fresh):,} historical trades from the "
                    "replay as a baseline. They are backtest results and were not "
                    "journalled as paper decisions."
                )
            self._clear_position()
            return []

        now = time.time()
        running = self.state.equity
        out: List[Decision] = []
        for t in fresh:
            running += t.pnl
            out.append(
                Decision(
                    bar_ts=bar.ts,
                    decided_at=now,
                    action="close",
                    direction=t.direction,
                    quantity=t.quantity,
                    price=t.exit_price,
                    equity=running,
                    r_multiple=t.r_multiple,
                    reason=f"{t.exit_reason}; {t.pnl:+,.2f} ({t.r_multiple:+.2f}R)",
                )
            )
        if fresh:
            self._clear_position()
        return out

    def _clear_position(self) -> None:
        st = self.state
        st.direction = FLAT
        st.quantity = 0.0
        st.entry_price = st.stop = st.target = st.entry_ts = 0.0

    def _adopt(self, target: OpenPosition) -> None:
        st = self.state
        st.direction = target.direction
        st.quantity = target.quantity
        st.entry_price = target.entry_price
        st.stop = target.stop
        st.target = target.target
        st.entry_ts = target.entry_ts

    # ------------------------------------------------------------------
    def _reconcile(self, target: Optional[OpenPosition], bar: Bar) -> List[Decision]:
        """Diff what the rules say we should hold against what we do hold."""
        now = time.time()
        st = self.state

        if target is None:
            if st.is_flat:
                return [
                    Decision(
                        bar_ts=bar.ts, decided_at=now, action="no_signal",
                        price=bar.close, equity=st.equity,
                        reason="flat, and the rules want flat",
                    )
                ]
            # The engine holds nothing but the book still does. _closures
            # normally handles this; reaching here means the position vanished
            # without a recorded trade, which is worth flagging loudly.
            held_dir, held_qty = st.direction, st.quantity
            self._clear_position()
            return [
                Decision(
                    bar_ts=bar.ts, decided_at=now, action="close",
                    direction=held_dir, quantity=held_qty, price=bar.close,
                    equity=st.equity,
                    reason="engine holds nothing but the book did; forced flat",
                )
            ]

        if st.is_flat:
            self._adopt(target)
            return [self._open_decision(target, bar, now, "engine opened a position")]

        same = (
            target.direction == st.direction
            and abs(target.entry_ts - st.entry_ts) <= AMEND_EPS
        )
        if not same:
            # A round trip completed and a new position opened between two
            # updates. Direction alone would have called this "hold".
            held_dir, held_qty = st.direction, st.quantity
            self._clear_position()
            self._adopt(target)
            return [
                Decision(
                    bar_ts=bar.ts, decided_at=now, action="close",
                    direction=held_dir, quantity=held_qty, price=bar.close,
                    equity=st.equity,
                    reason="replaced by a different position",
                ),
                self._open_decision(target, bar, now, "engine rotated into this"),
            ]

        changes = []
        if abs(target.stop - st.stop) > AMEND_EPS:
            changes.append(f"stop {st.stop:,.2f} -> {target.stop:,.2f}")
        if abs(target.target - st.target) > AMEND_EPS:
            changes.append(f"target {st.target:,.2f} -> {target.target:,.2f}")
        if abs(target.quantity - st.quantity) > EPS:
            changes.append(f"size {st.quantity:g} -> {target.quantity:g}")

        if changes:
            self._adopt(target)
            return [
                Decision(
                    bar_ts=bar.ts, decided_at=now, action="adjust",
                    direction=target.direction, quantity=target.quantity,
                    price=bar.close, stop=target.stop, target=target.target,
                    equity=st.equity, reason="; ".join(changes),
                )
            ]

        unreal = st.unrealised(bar.close, self.point_value)
        return [
            Decision(
                bar_ts=bar.ts, decided_at=now, action="hold",
                direction=st.direction, quantity=st.quantity, price=bar.close,
                stop=st.stop, target=st.target, equity=st.equity,
                reason=f"unchanged; unrealised {unreal:+,.2f}",
            )
        ]

    def _open_decision(
        self, target: OpenPosition, bar: Bar, now: float, why: str
    ) -> Decision:
        return Decision(
            bar_ts=bar.ts, decided_at=now,
            action="open_long" if target.direction > 0 else "open_short",
            direction=target.direction, quantity=target.quantity,
            price=target.entry_price, stop=target.stop, target=target.target,
            equity=self.state.equity, reason=why,
        )

    # ------------------------------------------------------------------
    def _journal(self, decision: Decision) -> None:
        append_jsonl(self.journal_path, decision)

    def read_journal(self) -> List[dict]:
        return read_jsonl(self.journal_path)

    # ------------------------------------------------------------------
    def status(self, last_price: Optional[float] = None) -> str:
        st = self.state
        when = (
            time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime(st.last_bar_ts))
            if st.last_bar_ts
            else "none yet"
        )
        pct = (
            (st.equity / self.cfg.starting_equity - 1.0) * 100.0
            if self.cfg.starting_equity
            else 0.0
        )
        lines = [
            "=" * 72,
            "Paper trading status",
            "=" * 72,
            f"Equity          {st.equity:,.2f}  ({pct:+.2f}% on "
            f"{self.cfg.starting_equity:,.0f})",
            f"Closed trades   {st.n_trades_seen}",
            f"Decisions       {st.n_decisions}",
            f"Bars replayed   {st.bars_seen:,}",
            f"Last bar        {when}",
        ]
        if st.is_flat:
            lines.append("Position        flat")
        else:
            side = "long" if st.direction > 0 else "short"
            lines.append(
                f"Position        {side} {st.quantity:g} @ {st.entry_price:,.2f} "
                f"(stop {st.stop:,.2f}, target {st.target:,.2f})"
            )
            if last_price:
                lines.append(
                    f"Unrealised      "
                    f"{st.unrealised(last_price, self.point_value):+,.2f}"
                )
        for w in self.warnings:
            lines.append(f"WARNING         {w}")
        lines += [
            "-" * 72,
            "No orders are placed. This records what the rules would have done,",
            "so that live fills can later be compared against what was assumed.",
            "=" * 72,
            "",
        ]
        return "\n".join(lines)
