"""Set and forget: a trend allocator built to run unattended for years.

This is deliberately not the order-flow strategy. That one needs tick data, has
a cost hurdle measured in half its own risk, and would need revalidating every
few months -- it is a high-touch system wearing a bot costume. What follows is
the opposite, and the two are meant to coexist: this is the core, that is a
satellite it has to earn its way into.

**The whole idea.** Hold spot BTC while price is above a long moving average.
Hold nothing otherwise. There is one number to choose and you will not tune it.
Returns come from being present for the multi-year advances and absent for the
multi-year declines; nothing here tries to be clever inside a week.

**Why binary exposure and not risk-based sizing.** A trend filter that risks
half a percent per signal and fires six times a year puts roughly nine tenths
of one percent of the account at stake annually. That is not an investment
strategy, it is a rounding error with a cron job. For a slow compounder the
position *is* the account: fully in, or fully out.

**Why spot and not perp.** Perp adds funding, which at a baseline 0.01% per
eight-hour settlement is 11.6% a year against a position held through the
trend, and adds liquidation, which converts a survivable drawdown into a
permanent loss. Neither risk buys anything a set-and-forget core wants. Run
the leveraged sleeve separately and consciously, if at all.

**The design principle behind every safety rail: a rail may block getting in,
never getting out.** Being kept out costs opportunity. Being kept in costs
money, and the moments a rail is most likely to misfire -- a data gap, a
violent bar, an outage -- are exactly the moments you most want the exit
available. So the minimum-hold and bad-data guards apply to risk-on
transitions only.

**What "set and forget" does not mean.** It does not mean unmonitored. It means
no discretionary decisions: nothing here asks you to have a view. You still
need an alarm that fires when the thing stops reporting, because the failure
mode of an unwatched bot is not a bad trade, it is silence while holding a
position nobody is looking at. ``status`` prints a staleness line for exactly
this reason.
"""

import time
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

from .data import Bar
from .footprint import EPS
from .persistence import append_jsonl, atomic_write_json, load_dataclass, read_jsonl

RISK_ON = "risk_on"
RISK_OFF = "risk_off"

#: Rails that may fire. Each one blocks a risk-on transition and nothing else.
RAIL_WARMUP = "warmup"
RAIL_MIN_HOLD = "min_hold"
RAIL_BAD_DATA = "bad_data"
RAIL_STALE = "stale_data"
RAIL_HALTED = "halted"


@dataclass(frozen=True)
class AutopilotConfig:
    """Everything the allocator can be told. Most of it you will never touch."""

    #: The one real parameter. Long enough that noise does not reach it.
    sma_period: int = 200

    #: Hysteresis. Price must close this far past the average to flip the
    #: regime, so a market oscillating around the line does not generate a
    #: trade per bar. This is the single most valuable knob after the period,
    #: because whipsaw, not trend accuracy, is what kills moving-average
    #: systems.
    band_pct: float = 0.03

    #: Bars that must pass before the allocator may buy again after selling.
    #: Applies to entries only -- an exit is never delayed.
    min_hold_bars: int = 5

    #: Fraction of equity committed when risk-on. Below 1.0 leaves a cash
    #: buffer at the cost of proportional upside.
    target_exposure: float = 1.0

    #: Skip a rebalance smaller than this fraction of equity. Below it the fee
    #: exceeds the tracking error being corrected.
    rebalance_threshold: float = 0.02

    #: Round-trip costs. Both are one-way fractions of notional.
    fee_rate: float = 0.001
    slippage_rate: float = 0.0005

    #: A bar older than this is treated as a data outage: the allocator holds
    #: whatever it holds and says so loudly. Default suits daily bars.
    max_bar_age_seconds: float = 60 * 60 * 36

    #: A close this far from the previous one is treated as suspect and blocks
    #: a *purchase*. It never blocks a sale.
    max_gap_pct: float = 0.35

    #: Equity this far below its peak halts the allocator until a human clears
    #: it. This is a bug and bad-data detector, not a risk control: set it
    #: wider than any drawdown the strategy legitimately produces, or it will
    #: fire at the bottom and lock you out of the recovery.
    #:
    #: The default is deliberately loose. A trend filter that stacks a few
    #: whipsaws and then rides one sharp decline down to its exit can print 40%
    #: without anything being wrong, and a threshold that trips on that is a
    #: worse problem than no threshold at all. Backtest first, then set this
    #: above the worst drawdown you see.
    halt_drawdown_pct: float = 0.60

    starting_equity: float = 10_000.0

    def __post_init__(self) -> None:
        if self.sma_period < 2:
            raise ValueError("sma_period must be at least 2")
        if not 0.0 <= self.band_pct < 1.0:
            raise ValueError("band_pct must be in [0, 1)")
        if self.min_hold_bars < 0:
            raise ValueError("min_hold_bars cannot be negative")
        if not 0.0 < self.target_exposure <= 1.0:
            raise ValueError("target_exposure must be in (0, 1]")
        if self.fee_rate < 0 or self.slippage_rate < 0:
            raise ValueError("costs cannot be negative")
        if not 0.0 < self.halt_drawdown_pct < 1.0:
            raise ValueError("halt_drawdown_pct must be in (0, 1)")
        if self.starting_equity <= 0:
            raise ValueError("starting_equity must be positive")

    @property
    def cost_rate(self) -> float:
        """One-way cost as a fraction of notional."""
        return self.fee_rate + self.slippage_rate

    def annual_cost_drag(self, flips_per_year: float) -> float:
        """Fraction of the account spent on fees at a given flip rate.

        Binary exposure means every flip trades the whole account, so this is
        larger than the same venue's drag on a risk-sized system. It is the
        number that decides whether a slower average is worth using.
        """
        return 2.0 * self.cost_rate * flips_per_year * self.target_exposure


@dataclass
class Verdict:
    """What the allocator wants on this bar, and why it wants it."""

    ts: float
    close: float
    regime: str
    target_exposure: float
    reason: str
    sma: Optional[float] = None
    #: Set when a rail overrode what the signal alone would have done.
    blocked_by: Optional[str] = None

    @property
    def wants_in(self) -> bool:
        return self.target_exposure > EPS


@dataclass
class Fill:
    """A trade the allocator would place. Nothing here sends it anywhere."""

    ts: float
    side: str  # buy | sell
    units: float
    price: float
    cost: float
    equity_after: float
    reason: str


@dataclass
class AutopilotState:
    """The book, plus everything needed to resume mid-trend after a restart."""

    cash: float = 0.0
    units: float = 0.0
    regime: str = RISK_OFF
    #: Bars processed since the regime last changed, for the minimum hold.
    #: A count rather than a bar index, so it keeps its meaning whether the
    #: caller passes the full history or only the bars since last time.
    bars_since_flip: int = 10_000
    peak_equity: float = 0.0
    halted: bool = False
    halt_reason: str = ""
    last_bar_ts: float = 0.0
    last_close: float = 0.0
    bars_seen: int = 0
    n_fills: int = 0
    started_at: float = 0.0

    def equity(self, price: float) -> float:
        return self.cash + self.units * price

    @property
    def is_flat(self) -> bool:
        return self.units <= EPS


class Autopilot:
    """The allocator. The same object backtests and runs live, by design.

    Two implementations of one rule set eventually disagree, and the
    disagreement always surfaces in production rather than in the test. So
    :meth:`decide` is the only place the rule lives, and both
    :meth:`backtest` and :meth:`update` call it.
    """

    def __init__(self, cfg: Optional[AutopilotConfig] = None):
        self.cfg = cfg or AutopilotConfig()

    # ------------------------------------------------------------------
    # The rule
    # ------------------------------------------------------------------
    def decide(
        self,
        bars: Sequence[Bar],
        index: int,
        regime: str,
        bars_since_flip: int,
        now: Optional[float] = None,
    ) -> Verdict:
        """What to hold at ``index``, given the regime carried in from before.

        Uses bars up to and including ``index`` and nothing after it, so
        replaying history through this produces the same answers it would have
        produced live.
        """
        cfg = self.cfg
        bar = bars[index]
        base = Verdict(
            ts=bar.ts, close=bar.close, regime=regime,
            target_exposure=cfg.target_exposure if regime == RISK_ON else 0.0,
            reason="regime unchanged",
        )

        if index < cfg.sma_period - 1:
            base.regime = RISK_OFF
            base.target_exposure = 0.0
            base.blocked_by = RAIL_WARMUP
            base.reason = (
                f"bar {index + 1} of the {cfg.sma_period} needed before the "
                "average means anything"
            )
            return base

        avg = self._average(bars, index)
        base.sma = avg
        upper = avg * (1.0 + cfg.band_pct)
        lower = avg * (1.0 - cfg.band_pct)

        # ---- Exit side. Nothing may block this. ----
        if regime == RISK_ON:
            if bar.close < lower:
                base.regime = RISK_OFF
                base.target_exposure = 0.0
                base.reason = (
                    f"close {bar.close:,.2f} below the {cfg.band_pct:.0%} band "
                    f"under SMA{cfg.sma_period} {avg:,.2f}"
                )
            else:
                base.reason = (
                    f"holding; close {bar.close:,.2f} vs SMA{cfg.sma_period} "
                    f"{avg:,.2f}"
                )
            return base

        # ---- Entry side. Rails apply here. ----
        if bar.close <= upper:
            base.reason = (
                f"out; close {bar.close:,.2f} has not cleared "
                f"{upper:,.2f} ({cfg.band_pct:.0%} above SMA{cfg.sma_period})"
            )
            return base

        blocked = self._entry_rail(bars, index, bars_since_flip, now)
        if blocked is not None:
            base.blocked_by, base.reason = blocked
            return base

        base.regime = RISK_ON
        base.target_exposure = cfg.target_exposure
        base.reason = (
            f"close {bar.close:,.2f} cleared {upper:,.2f} "
            f"({cfg.band_pct:.0%} above SMA{cfg.sma_period} {avg:,.2f})"
        )
        return base

    def _entry_rail(
        self,
        bars: Sequence[Bar],
        index: int,
        bars_since_flip: int,
        now: Optional[float],
    ):
        """The rails, in order. Each returns ``(rail, reason)`` or ``None``."""
        cfg = self.cfg
        bar = bars[index]

        if bars_since_flip < cfg.min_hold_bars:
            return (
                RAIL_MIN_HOLD,
                f"signal is on, but only {bars_since_flip} of "
                f"{cfg.min_hold_bars} bars since the last flip -- refusing "
                "to churn",
            )

        if index > 0:
            prev = bars[index - 1].close
            if prev > EPS:
                move = abs(bar.close - prev) / prev
                if move > cfg.max_gap_pct:
                    return (
                        RAIL_BAD_DATA,
                        f"close moved {move:.1%} in one bar, over the "
                        f"{cfg.max_gap_pct:.0%} limit -- treating as suspect "
                        "data and not buying into it",
                    )

        if now is not None:
            age = now - bar.ts
            if age > cfg.max_bar_age_seconds:
                return (
                    RAIL_STALE,
                    f"newest bar is {age / 3600:.1f}h old, over the "
                    f"{cfg.max_bar_age_seconds / 3600:.0f}h limit -- the feed "
                    "is behind, not buying blind",
                )
        return None

    def _average(self, bars: Sequence[Bar], index: int) -> float:
        lo = index - self.cfg.sma_period + 1
        window = bars[lo : index + 1]
        return sum(b.close for b in window) / len(window)

    # ------------------------------------------------------------------
    # Book keeping
    # ------------------------------------------------------------------
    def _rebalance(
        self, state: AutopilotState, verdict: Verdict, index: int
    ) -> Optional[Fill]:
        """Move the book toward the target. Returns the fill, or None."""
        cfg = self.cfg
        price = verdict.close
        if price <= EPS:
            return None

        equity = state.equity(price)
        want_units = (verdict.target_exposure * equity) / price
        delta = want_units - state.units
        if abs(delta) * price < cfg.rebalance_threshold * max(equity, EPS):
            return None

        # Buying spends cash on units and the fee; selling returns cash net of
        # it. Sizing off pre-fee equity slightly overshoots on the way in,
        # which is corrected on the next bar and never leaves negative cash at
        # these cost levels.
        cost = abs(delta) * price * cfg.cost_rate
        if delta > 0:
            spend = delta * price + cost
            if spend > state.cash:
                delta = max(0.0, state.cash / (price * (1.0 + cfg.cost_rate)))
                if delta * price < cfg.rebalance_threshold * max(equity, EPS):
                    return None
                cost = delta * price * cfg.cost_rate
                spend = delta * price + cost
            state.cash -= spend
            state.units += delta
            side = "buy"
        else:
            qty = min(-delta, state.units)
            if qty <= EPS:
                return None
            cost = qty * price * cfg.cost_rate
            state.cash += qty * price - cost
            state.units -= qty
            delta = -qty
            side = "sell"

        state.n_fills += 1
        return Fill(
            ts=verdict.ts, side=side, units=abs(delta), price=price, cost=cost,
            equity_after=state.equity(price), reason=verdict.reason,
        )

    def _apply(
        self, state: AutopilotState, verdict: Verdict, index: int
    ) -> Optional[Fill]:
        """One bar's worth of state transition, rails included."""
        cfg = self.cfg
        price = verdict.close

        if state.halted:
            verdict.regime = RISK_OFF
            verdict.target_exposure = 0.0
            verdict.blocked_by = RAIL_HALTED
            verdict.reason = f"halted: {state.halt_reason}"

        if verdict.regime != state.regime:
            state.regime = verdict.regime
            state.bars_since_flip = 0
        else:
            state.bars_since_flip += 1

        fill = self._rebalance(state, verdict, index)

        equity = state.equity(price)
        state.peak_equity = max(state.peak_equity, equity)
        state.last_bar_ts = verdict.ts
        state.last_close = price

        if (
            not state.halted
            and state.peak_equity > EPS
            and equity < state.peak_equity * (1.0 - cfg.halt_drawdown_pct)
        ):
            state.halted = True
            state.halt_reason = (
                f"equity {equity:,.2f} is {1 - equity / state.peak_equity:.1%} "
                f"below the peak of {state.peak_equity:,.2f}, past the "
                f"{cfg.halt_drawdown_pct:.0%} limit. This is a bug and "
                "bad-data alarm, not a stop loss. Investigate before clearing."
            )
        return fill

    # ------------------------------------------------------------------
    def backtest(self, bars: Sequence[Bar]) -> "AutopilotResult":
        """Replay the rule over history using the live code path."""
        if not bars:
            raise ValueError("no bars")
        state = AutopilotState(
            cash=self.cfg.starting_equity,
            peak_equity=self.cfg.starting_equity,
            started_at=time.time(),
        )
        curve: List[float] = []
        fills: List[Fill] = []
        verdicts: List[Verdict] = []

        for i in range(len(bars)):
            v = self.decide(bars, i, state.regime, state.bars_since_flip)
            fill = self._apply(state, v, i)
            if fill is not None:
                fills.append(fill)
            verdicts.append(v)
            curve.append(state.equity(bars[i].close))
            state.bars_seen = i + 1

        return AutopilotResult(
            cfg=self.cfg, state=state, equity_curve=curve,
            bar_timestamps=[b.ts for b in bars], fills=fills, verdicts=verdicts,
            closes=[b.close for b in bars],
        )


@dataclass
class AutopilotResult:
    cfg: AutopilotConfig
    state: AutopilotState
    equity_curve: List[float]
    bar_timestamps: List[float]
    fills: List[Fill]
    verdicts: List[Verdict]
    closes: List[float] = field(default_factory=list)

    @property
    def final_equity(self) -> float:
        return self.equity_curve[-1] if self.equity_curve else self.cfg.starting_equity

    @property
    def total_return(self) -> float:
        return self.final_equity / self.cfg.starting_equity - 1.0

    @property
    def n_flips(self) -> int:
        """Regime changes, which is what actually drives the fee bill."""
        flips = 0
        prev = RISK_OFF
        for v in self.verdicts:
            if v.regime != prev:
                flips += 1
                prev = v.regime
        return flips

    @property
    def years(self) -> float:
        if len(self.bar_timestamps) < 2:
            return 0.0
        span = self.bar_timestamps[-1] - self.bar_timestamps[0]
        return span / (365.25 * 24 * 3600)

    @property
    def cagr(self) -> Optional[float]:
        if self.years <= 0 or self.final_equity <= 0:
            return None
        return (self.final_equity / self.cfg.starting_equity) ** (1 / self.years) - 1

    @property
    def max_drawdown(self) -> float:
        peak, worst = EPS, 0.0
        for e in self.equity_curve:
            peak = max(peak, e)
            worst = max(worst, 1.0 - e / peak)
        return worst

    @property
    def total_costs(self) -> float:
        return sum(f.cost for f in self.fills)

    @property
    def time_in_market(self) -> float:
        if not self.verdicts:
            return 0.0
        return sum(1 for v in self.verdicts if v.wants_in) / len(self.verdicts)

    @property
    def bars_halted(self) -> int:
        """Bars spent locked flat by the drawdown halt.

        Any number above zero means the headline return describes a bot that
        stopped trading partway through, which is a different thing from the
        strategy's result and must never be reported as if it were the same.
        """
        return sum(1 for v in self.verdicts if v.blocked_by == RAIL_HALTED)

    def buy_and_hold(self) -> Optional[float]:
        """The benchmark. Beating it on return is not the point, but you must
        know by how much you are not, because the gap is what the drawdown
        reduction costs."""
        if len(self.closes) < 2 or self.closes[0] <= EPS:
            return None
        return self.closes[-1] / self.closes[0] - 1.0

    def buy_and_hold_drawdown(self) -> float:
        peak, worst = EPS, 0.0
        for c in self.closes:
            peak = max(peak, c)
            worst = max(worst, 1.0 - c / peak)
        return worst


def format_autopilot(result: AutopilotResult, title: str = "Autopilot") -> str:
    """The numbers that decide whether this is worth running.

    Return alone answers nothing. A trend filter on a rising asset almost
    always returns less than holding it; what it buys is a smaller worst case
    and a shorter time to recover. Both sides are printed together so the
    trade being made is visible rather than assumed.
    """
    cfg = result.cfg
    bh = result.buy_and_hold()
    cagr = result.cagr
    rows = [
        ("Total return", f"{result.total_return:+.1%}",
         f"{bh:+.1%}" if bh is not None else "n/a"),
        ("CAGR", f"{cagr:+.1%}" if cagr is not None else "n/a", ""),
        ("Max drawdown", f"{result.max_drawdown:.1%}",
         f"{result.buy_and_hold_drawdown():.1%}"),
        ("Time in market", f"{result.time_in_market:.0%}", "100%"),
        ("Regime flips", f"{result.n_flips}", "0"),
        ("Fills", f"{len(result.fills)}", "1"),
        ("Costs paid", f"{result.total_costs:,.2f}", ""),
    ]

    out = [
        "=" * 72,
        f"{title} -- SMA{cfg.sma_period}, {cfg.band_pct:.0%} band, "
        f"{cfg.target_exposure:.0%} exposure",
        "=" * 72,
        f"{'':<18}{'autopilot':>14}{'buy and hold':>16}",
        "-" * 72,
    ]
    for label, a, b in rows:
        out.append(f"{label:<18}{a:>14}{b:>16}")

    out += ["-" * 72]
    if result.bars_halted:
        share = result.bars_halted / max(len(result.verdicts), 1)
        out += [
            "*** THIS RUN HALTED ***",
            f"The drawdown halt fired and locked the book flat for "
            f"{result.bars_halted:,} of {len(result.verdicts):,} bars "
            f"({share:.0%} of the sample).",
            "Every number above therefore describes a bot that stopped trading",
            f"partway through, not the strategy. Raise --halt-drawdown above the",
            f"{result.max_drawdown:.0%} drawdown seen here and run it again.",
            "-" * 72,
        ]
    if result.years > 0:
        flips_pa = result.n_flips / result.years
        out.append(
            f"{flips_pa:.1f} regime flips a year over {result.years:.1f} years. "
            f"At {cfg.cost_rate:.2%} per side"
        )
        out.append(
            f"that is {cfg.annual_cost_drag(flips_pa):.2%} of the account a year "
            "in fees and slippage."
        )
    if bh is not None and result.total_return < bh:
        out.append(
            f"Underperformed holding by {bh - result.total_return:.1%}, and cut the "
            f"worst drawdown from {result.buy_and_hold_drawdown():.0%} to "
            f"{result.max_drawdown:.0%}."
        )
        out.append("That exchange is the entire proposition. Decide if you want it.")
    elif bh is not None:
        out.append(
            f"Beat holding by {result.total_return - bh:.1%} with a smaller worst "
            "case. Treat one sample this good with suspicion, not delight."
        )
    out += ["=" * 72, ""]
    return "\n".join(out)


class AutopilotRunner:
    """Runs the allocator on a schedule, keeping its book on disk.

    Unlike the order-flow paper trader, this does not replay history on every
    invocation. It does not need to: the whole of the allocator's memory is
    the regime, the bars since it changed, and the book, all of which fit in
    the state file. New bars are folded in one at a time.

    The first run is different. It replays the history to work out which
    regime the market is *currently* in, then makes one trade to match. That
    is what deploying into an existing trend actually looks like, and it is
    honest in a way that pretending to have traded the whole history is not.
    """

    def __init__(
        self,
        cfg: Optional[AutopilotConfig] = None,
        state_path: str = "autopilot/state.json",
        journal_path: str = "autopilot/journal.jsonl",
    ):
        self.cfg = cfg or AutopilotConfig()
        self.pilot = Autopilot(self.cfg)
        self.state_path = state_path
        self.journal_path = journal_path
        loaded = load_dataclass(state_path, AutopilotState, what="autopilot state")
        self.state = loaded or AutopilotState(
            cash=self.cfg.starting_equity,
            peak_equity=self.cfg.starting_equity,
            started_at=time.time(),
        )
        self.warnings: List[str] = []

    # ------------------------------------------------------------------
    def update(
        self, bars: Sequence[Bar], now: Optional[float] = None
    ) -> List[Verdict]:
        """Fold in any bars newer than the last one processed.

        Returns the verdicts taken, newest last. An empty list means there was
        nothing new, which is the usual answer and the reason this is safe to
        run on a tight schedule.
        """
        self.warnings = []
        now = time.time() if now is None else now
        if not bars:
            return []
        if len(bars) < self.cfg.sma_period:
            raise ValueError(
                f"need at least {self.cfg.sma_period} bars for an "
                f"SMA{self.cfg.sma_period}, got {len(bars)}"
            )
        if bars[-1].ts <= self.state.last_bar_ts:
            return []

        first_run = self.state.bars_seen == 0
        if first_run:
            self._bootstrap(bars)
            fresh = [len(bars) - 1]
        else:
            fresh = [i for i, b in enumerate(bars) if b.ts > self.state.last_bar_ts]
            skipped = len(fresh) - 1
            if skipped > 0:
                self.warnings.append(
                    f"{skipped} bar(s) arrived at once. The allocator caught up "
                    "on all of them, but a daily bot seeing this is behind "
                    "schedule -- check the feed."
                )

        out: List[Verdict] = []
        for i in fresh:
            v = self.pilot.decide(
                bars, i, self.state.regime, self.state.bars_since_flip, now=now
            )
            fill = self.pilot._apply(self.state, v, i)
            self.state.bars_seen += 1
            self._journal(v, fill)
            out.append(v)
            if v.blocked_by in (RAIL_STALE, RAIL_BAD_DATA):
                self.warnings.append(v.reason)

        if self.state.halted:
            self.warnings.append(self.state.halt_reason)
        atomic_write_json(self.state_path, self.state)
        return out

    def _bootstrap(self, bars: Sequence[Bar]) -> None:
        """Establish the current regime from history without trading it."""
        regime, since = RISK_OFF, 10_000
        for i in range(len(bars) - 1):
            v = self.pilot.decide(bars, i, regime, since)
            if v.regime != regime:
                regime, since = v.regime, 0
            else:
                since += 1
        self.state.regime = regime
        self.state.bars_since_flip = since
        self.warnings.append(
            f"first run: replayed {len(bars):,} bars to establish the regime "
            f"({regime}) without trading them. The book starts flat with "
            f"{self.cfg.starting_equity:,.2f} and takes one trade to match."
        )

    def _journal(self, verdict: Verdict, fill: Optional[Fill]) -> None:
        record = {
            "bar_ts": verdict.ts,
            "decided_at": time.time(),
            "close": verdict.close,
            "regime": verdict.regime,
            "target_exposure": verdict.target_exposure,
            "sma": verdict.sma,
            "blocked_by": verdict.blocked_by,
            "reason": verdict.reason,
            "equity": self.state.equity(verdict.close),
            "units": self.state.units,
            "cash": self.state.cash,
            "fill": (
                {
                    "side": fill.side, "units": fill.units,
                    "price": fill.price, "cost": fill.cost,
                }
                if fill
                else None
            ),
        }
        append_jsonl(self.journal_path, record)

    def read_journal(self) -> List[dict]:
        return read_jsonl(self.journal_path)

    def clear_halt(self) -> None:
        """Resume after a halt. Deliberately a separate, manual action."""
        self.state.halted = False
        self.state.halt_reason = ""
        self.state.peak_equity = self.state.equity(self.state.last_close)
        atomic_write_json(self.state_path, self.state)

    # ------------------------------------------------------------------
    def status(self, now: Optional[float] = None) -> str:
        st, cfg = self.state, self.cfg
        now = time.time() if now is None else now
        price = st.last_close
        equity = st.equity(price) if price else st.cash
        pct = (equity / cfg.starting_equity - 1.0) * 100.0 if cfg.starting_equity else 0.0
        age_h = (now - st.last_bar_ts) / 3600.0 if st.last_bar_ts else None

        lines = [
            "=" * 72,
            "Autopilot status",
            "=" * 72,
            f"Regime          {st.regime}",
            f"Equity          {equity:,.2f}  ({pct:+.2f}% on "
            f"{cfg.starting_equity:,.0f})",
            f"Holding         {st.units:.8f} units @ last close {price:,.2f}",
            f"Cash            {st.cash:,.2f}",
            f"Fills           {st.n_fills} over {st.bars_seen:,} bars",
            f"Bars since flip {st.bars_since_flip}",
        ]
        if age_h is None:
            lines.append("Data age        no bars processed yet")
        else:
            stale = age_h * 3600 > cfg.max_bar_age_seconds
            lines.append(
                f"Data age        {age_h:.1f}h"
                + ("   <-- STALE, check the feed" if stale else "")
            )
        if st.halted:
            lines += ["", "*** HALTED ***", st.halt_reason,
                      "Clear it with --clear-halt once you know why."]
        for w in self.warnings:
            lines.append(f"WARNING         {w}")
        lines += [
            "-" * 72,
            "No orders are placed. Set-and-forget means no discretionary",
            "decisions, not no supervision: alarm on the data age above.",
            "=" * 72,
            "",
        ]
        return "\n".join(lines)
