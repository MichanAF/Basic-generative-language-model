"""Bars, trades, CSV loading, and a synthetic tape for testing the mechanics."""

import csv
import math
import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence

from .footprint import EPS, Footprint

BUY_WORDS = {"b", "buy", "bid", "a", "ask", "1", "+1", "up", "aggressive_buy", "taker_buy"}
SELL_WORDS = {"s", "sell", "-1", "down", "aggressive_sell", "taker_sell"}


@dataclass(frozen=True)
class Trade:
    """A single print off the tape.

    ``aggressor`` is ``+1`` when a market buy lifted the offer and ``-1`` when a
    market sell hit the bid. Without this field there is no order flow, only
    volume -- see :func:`Footprint.from_ohlcv_proxy`.
    """

    ts: float
    price: float
    size: float
    aggressor: int


@dataclass
class Bar:
    ts: float
    open: float
    high: float
    low: float
    close: float
    volume: float
    footprint: Footprint

    # -- geometry ------------------------------------------------------
    @property
    def range(self) -> float:
        return self.high - self.low

    @property
    def body(self) -> float:
        return abs(self.close - self.open)

    @property
    def body_frac(self) -> float:
        return self.body / self.range if self.range > EPS else 0.0

    @property
    def upper_wick(self) -> float:
        return self.high - max(self.open, self.close)

    @property
    def lower_wick(self) -> float:
        return min(self.open, self.close) - self.low

    @property
    def upper_wick_frac(self) -> float:
        return self.upper_wick / self.range if self.range > EPS else 0.0

    @property
    def lower_wick_frac(self) -> float:
        return self.lower_wick / self.range if self.range > EPS else 0.0

    @property
    def close_location(self) -> float:
        """Where the close sits in the range: 0.0 at the low, 1.0 at the high."""
        return (self.close - self.low) / self.range if self.range > EPS else 0.5

    @property
    def is_up(self) -> bool:
        return self.close > self.open

    # -- flow ----------------------------------------------------------
    @property
    def delta(self) -> float:
        return self.footprint.delta

    @property
    def normalized_delta(self) -> float:
        return self.footprint.normalized_delta

    def price_at_frac(self, frac: float) -> float:
        """Price ``frac`` of the way up the bar's range."""
        return self.low + frac * self.range


# ----------------------------------------------------------------------
# Aggregation
# ----------------------------------------------------------------------
def bars_from_trades(
    trades: Sequence[Trade], timeframe_seconds: float, tick_size: float
) -> List[Bar]:
    """Aggregate a trade tape into time bars with true footprints."""
    if timeframe_seconds <= 0:
        raise ValueError("timeframe_seconds must be positive")

    bars: List[Bar] = []
    bucket: Optional[float] = None
    cur: Optional[Bar] = None

    for t in trades:
        b = (t.ts // timeframe_seconds) * timeframe_seconds
        if cur is None or b != bucket:
            if cur is not None:
                bars.append(cur)
            bucket = b
            cur = Bar(
                ts=b,
                open=t.price,
                high=t.price,
                low=t.price,
                close=t.price,
                volume=0.0,
                footprint=Footprint(tick_size=tick_size),
            )
        cur.high = max(cur.high, t.price)
        cur.low = min(cur.low, t.price)
        cur.close = t.price
        cur.volume += t.size
        cur.footprint.add(t.price, t.size, t.aggressor)

    if cur is not None:
        bars.append(cur)
    return bars


# ----------------------------------------------------------------------
# CSV loading
# ----------------------------------------------------------------------
def _parse_side(raw: str) -> int:
    s = str(raw).strip().lower()
    if s in BUY_WORDS:
        return 1
    if s in SELL_WORDS:
        return -1
    try:  # numeric encodings
        return 1 if float(s) > 0 else -1
    except ValueError as exc:
        raise ValueError(f"unrecognized aggressor side: {raw!r}") from exc


def _pick(row: Dict[str, str], *names: str) -> Optional[str]:
    lowered = {k.strip().lower(): v for k, v in row.items() if k}
    for n in names:
        if n in lowered and lowered[n] not in ("", None):
            return lowered[n]
    return None


def load_trades_csv(path: str) -> List[Trade]:
    """Load a tick tape.

    Expected columns (case insensitive, aliases accepted):
    ``ts|time|timestamp``, ``price``, ``size|qty|volume``,
    ``side|aggressor|taker_side``.
    """
    out: List[Trade] = []
    with open(path, newline="", encoding="utf-8") as fh:
        for lineno, row in enumerate(csv.DictReader(fh), start=2):
            ts = _pick(row, "ts", "time", "timestamp", "datetime")
            price = _pick(row, "price", "px", "last")
            size = _pick(row, "size", "qty", "quantity", "volume", "vol")
            side = _pick(row, "side", "aggressor", "taker_side", "direction")
            if ts is None or price is None or size is None or side is None:
                raise ValueError(f"{path}:{lineno}: missing a required trade column")
            out.append(
                Trade(
                    ts=float(ts),
                    price=float(price),
                    size=float(size),
                    aggressor=_parse_side(side),
                )
            )
    out.sort(key=lambda t: t.ts)
    return out


def load_bars_csv(path: str, tick_size: float) -> List[Bar]:
    """Load OHLCV bars.

    Required: ``ts``, ``open``, ``high``, ``low``, ``close``, ``volume``.
    Optional: ``ask_volume``/``buy_volume`` and ``bid_volume``/``sell_volume``,
    or a ``delta`` column. When present the real aggressor split is used and
    only the distribution of volume across price is estimated; when absent the
    split itself is estimated from the close location, which is materially
    weaker -- the resulting footprints are flagged as proxies.
    """
    bars: List[Bar] = []
    with open(path, newline="", encoding="utf-8") as fh:
        for lineno, row in enumerate(csv.DictReader(fh), start=2):
            try:
                ts = float(_pick(row, "ts", "time", "timestamp", "datetime"))
                o = float(_pick(row, "open", "o"))
                h = float(_pick(row, "high", "h"))
                l = float(_pick(row, "low", "l"))
                c = float(_pick(row, "close", "c"))
                v = float(_pick(row, "volume", "vol", "v"))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{path}:{lineno}: bad or missing OHLCV column") from exc

            ask_raw = _pick(row, "ask_volume", "buy_volume", "taker_buy_volume")
            bid_raw = _pick(row, "bid_volume", "sell_volume", "taker_sell_volume")
            delta_raw = _pick(row, "delta")

            buy_frac: Optional[float] = None
            if ask_raw is not None and bid_raw is not None:
                a, b = float(ask_raw), float(bid_raw)
                if a + b > EPS:
                    buy_frac = a / (a + b)
            elif delta_raw is not None and v > EPS:
                # delta = ask - bid, volume = ask + bid  =>  ask/volume
                buy_frac = min(1.0, max(0.0, 0.5 * (1.0 + float(delta_raw) / v)))

            fp = Footprint.from_ohlcv_proxy(o, h, l, c, v, tick_size, buy_frac=buy_frac)
            bars.append(Bar(ts=ts, open=o, high=h, low=l, close=c, volume=v, footprint=fp))
    bars.sort(key=lambda b: b.ts)
    return bars


def write_bars_csv(path: str, bars: Iterable[Bar]) -> None:
    """Dump bars (with measured aggressor volumes) for reuse."""
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["ts", "open", "high", "low", "close", "volume", "ask_volume", "bid_volume"])
        for b in bars:
            w.writerow(
                [
                    f"{b.ts:.0f}",
                    f"{b.open:.6f}",
                    f"{b.high:.6f}",
                    f"{b.low:.6f}",
                    f"{b.close:.6f}",
                    f"{b.volume:.4f}",
                    f"{sum(b.footprint.ask_volume.values()):.4f}",
                    f"{sum(b.footprint.bid_volume.values()):.4f}",
                ]
            )


# ----------------------------------------------------------------------
# Synthetic tape
# ----------------------------------------------------------------------
@dataclass
class _Wall:
    """A resting passive block: the thing that actually creates resistance."""

    level: int
    side: int  # +1 sell wall (resistance), -1 buy wall (support)
    capacity: float
    ttl: int
    absorbed: float = 0.0


@dataclass
class SyntheticConfig:
    n_ticks: int = 240_000
    tick_size: float = 0.25
    start_price: float = 4500.0
    #: Probability per tick that price attempts a move rather than trading flat.
    move_prob: float = 0.55
    #: Probability the trend regime flips on any given tick.
    regime_flip_prob: float = 0.0008
    trend_strength: float = 0.06
    mean_size: float = 3.0
    #: Probability per tick of spawning a new passive wall.
    wall_spawn_prob: float = 0.0009
    wall_distance_ticks: int = 14
    wall_capacity_mean: float = 900.0
    wall_capacity_spread: float = 0.9
    wall_ttl: int = 4_000
    #: Push-off applied once a wall wins the fight, in ticks per tick of tape.
    wall_reject_strength: float = 0.35
    seconds_per_tick: float = 1.5
    seed: int = 7


def synthetic_trades(cfg: Optional[SyntheticConfig] = None) -> List[Trade]:
    """Generate a tape that contains genuine absorption events.

    The generator plants resting passive blocks at random distances from price.
    When price reaches one, aggressive orders trade into it without moving
    price -- real absorption, with the aggressor tags to match. If the block's
    capacity is exhausted first, price breaks through instead.

    Capacity is drawn from a wide lognormal-ish distribution, so a meaningful
    share of walls break. This is deliberate: a tape where every level holds
    would make any reversal strategy look brilliant. Even so, treat results on
    this data as a test that the *code* works, never as evidence of an edge --
    the generator has no news, no participants, and no reflexivity.
    """
    cfg = cfg or SyntheticConfig()
    rng = random.Random(cfg.seed)

    level = int(round(cfg.start_price / cfg.tick_size))
    trend = 0
    walls: List[_Wall] = []
    trades: List[Trade] = []
    ts = 0.0
    push = 0.0  # residual rejection momentum after a wall defends

    for _ in range(cfg.n_ticks):
        ts += cfg.seconds_per_tick * rng.uniform(0.4, 1.6)

        if rng.random() < cfg.regime_flip_prob:
            trend = rng.choice([-1, 0, 0, 1])

        if rng.random() < cfg.wall_spawn_prob and len(walls) < 6:
            side = rng.choice([1, -1])
            dist = rng.randint(cfg.wall_distance_ticks // 2, cfg.wall_distance_ticks * 2)
            walls.append(
                _Wall(
                    level=level + side * dist,
                    side=side,
                    capacity=cfg.wall_capacity_mean
                    * (1.0 + rng.uniform(-cfg.wall_capacity_spread, cfg.wall_capacity_spread)),
                    ttl=cfg.wall_ttl,
                )
            )

        for w in walls:
            w.ttl -= 1
        walls = [w for w in walls if w.ttl > 0 and w.capacity > 0]

        size = max(1.0, round(rng.expovariate(1.0 / cfg.mean_size), 2))

        drift = trend * cfg.trend_strength + push
        push *= 0.985
        step = 0
        if rng.random() < cfg.move_prob + abs(drift) * 0.5:
            up_prob = 0.5 + drift * 0.5
            step = 1 if rng.random() < min(0.95, max(0.05, up_prob)) else -1

        aggressor = step if step != 0 else rng.choice([1, -1])
        target = level + step

        blocking = next(
            (
                w
                for w in walls
                if (w.side == 1 and step > 0 and target >= w.level)
                or (w.side == -1 and step < 0 and target <= w.level)
            ),
            None,
        )

        if blocking is not None:
            # Aggressors trade into the passive block; price does not advance.
            blocking.capacity -= size
            blocking.absorbed += size
            trades.append(
                Trade(ts=ts, price=blocking.level * cfg.tick_size, size=size, aggressor=aggressor)
            )
            level = blocking.level
            if blocking.capacity <= 0:  # the wall got run over -> breakout
                walls.remove(blocking)
                push += blocking.side * cfg.wall_reject_strength
            else:  # the wall defends -> rejection away from it
                push -= blocking.side * cfg.wall_reject_strength * 0.35
            continue

        level = target
        trades.append(Trade(ts=ts, price=level * cfg.tick_size, size=size, aggressor=aggressor))

    return trades


def synthetic_bars(
    timeframe_seconds: float = 300.0, cfg: Optional[SyntheticConfig] = None
) -> List[Bar]:
    """Convenience wrapper: synthetic tape aggregated into bars."""
    cfg = cfg or SyntheticConfig()
    return bars_from_trades(synthetic_trades(cfg), timeframe_seconds, cfg.tick_size)


def synthetic_daily_bars(
    n_bars: int = 1_500,
    seed: int = 7,
    start_price: float = 10_000.0,
    bar_seconds: float = 86_400.0,
    tick_size: float = 10.0,
    start_ts: float = 0.0,
) -> List[Bar]:
    """A regime-switching random walk, for exercising slow allocators.

    The tick generator cannot practically produce years of daily bars -- a
    thousand of them would need tens of millions of prints. This produces the
    price path directly instead.

    It alternates between bull, bear and directionless regimes with persistent
    drift, because that is the structure a trend filter exists to exploit and a
    plain random walk contains none of it. That also makes it worthless as
    evidence: a filter finding regimes that were deliberately planted proves
    the code runs, nothing more. Footprints are proxies and carry no order-flow
    information, so do not point the order-flow engine at this.
    """
    if n_bars < 1:
        raise ValueError("n_bars must be positive")
    if start_price <= 0:
        raise ValueError("start_price must be positive")

    rnd = random.Random(seed)
    # (daily drift, daily volatility, expected regime length in bars)
    regimes = ((0.0030, 0.030, 260), (-0.0035, 0.035, 150), (0.0000, 0.025, 120))

    price = start_price
    bars: List[Bar] = []
    drift, vol, mean_len = regimes[0]
    left = mean_len

    for i in range(n_bars):
        if left <= 0:
            drift, vol, mean_len = regimes[rnd.randrange(len(regimes))]
            left = max(20, int(rnd.expovariate(1.0 / mean_len)))
        left -= 1

        open_ = price
        price = max(price * math.exp(rnd.gauss(drift, vol)), tick_size)
        high = max(open_, price) * (1.0 + abs(rnd.gauss(0.0, vol / 3)))
        low = min(open_, price) * (1.0 - abs(rnd.gauss(0.0, vol / 3)))
        volume = max(1.0, rnd.lognormvariate(6.0, 0.4))
        bars.append(
            Bar(
                ts=start_ts + i * bar_seconds,
                open=open_,
                high=high,
                low=max(low, tick_size),
                close=price,
                volume=volume,
                footprint=Footprint.from_ohlcv_proxy(
                    open_=open_, high=high, low=max(low, tick_size), close=price,
                    volume=volume, tick_size=tick_size,
                ),
            )
        )
    return bars
