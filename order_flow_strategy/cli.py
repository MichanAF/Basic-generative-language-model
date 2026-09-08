"""Command line entry point.

    python -m order_flow_strategy demo
    python -m order_flow_strategy backtest --trades ticks.csv --timeframe 300
    python -m order_flow_strategy backtest --binance-klines data/btc --preset btc
    python -m order_flow_strategy optimize --bars bars.csv --folds 4
    python -m order_flow_strategy scan --trades ticks.csv --timeframe 300

Fetching Binance archives is a separate step, since it is the only part that
needs the network:

    python -m order_flow_strategy.sources.fetch --symbol BTCUSDT --interval 5m \
        --start 2023-08 --end 2025-07 --out data/btc
"""

import argparse
import csv
import sys
from typing import List, Optional, Sequence

from .backtest import Backtester
from .config import ENTRY_MODES, REGIME_FILTERS, StrategyConfig
from .data import (
    Bar,
    SyntheticConfig,
    bars_from_trades,
    load_bars_csv,
    load_trades_csv,
    synthetic_trades,
)
from .metrics import summarize
from .optimize import DEFAULT_GRID, FAST_GRID, format_recommendation, walk_forward
from .presets import PRESETS
from .presets import build as build_preset
from .sources import load_binance_paths
from .baselines import buy_and_hold, trend_baseline
from .confluence import analyse as analyse_confluence
from .confluence import format_confluence
from .reporting_html import build_report
from .funding import FundingSchedule
from .report import (
    DISCLAIMER,
    format_comparison,
    format_levels,
    format_signal,
    format_stats,
    format_trades,
)
from .signals import SignalEngine, prepare


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="order_flow_strategy",
        description="Order-flow resistance / two-candle reversal strategy.",
        epilog=DISCLAIMER,
    )
    sub = p.add_subparsers(dest="command", required=True)

    for name, help_text in (
        ("backtest", "Run the strategy over historical data."),
        ("optimize", "Walk-forward parameter search and recommendations."),
        ("scan", "Show live levels and any setup on the most recent bar."),
        ("demo", "Generate a synthetic tape and run everything on it."),
        ("compare", "Run the strategy against trend-following and buy-and-hold."),
        ("report", "Write a self-contained HTML report of the whole comparison."),
    ):
        s = sub.add_parser(name, help=help_text, description=help_text)
        _add_data_args(s)
        _add_strategy_args(s)
        if name == "optimize":
            s.add_argument("--folds", type=int, default=4, help="Walk-forward folds (default 4).")
            s.add_argument(
                "--fast", action="store_true", help="Use the smaller, quicker parameter grid."
            )
            s.add_argument(
                "--min-trades",
                type=int,
                default=15,
                help="Trades a fold must produce to be rankable (default 15).",
            )
        if name in ("backtest", "demo", "compare", "report"):
            s.add_argument(
                "--show-trades", type=int, default=20, help="Trade rows to print (0 for all)."
            )
            s.add_argument("--export-trades", metavar="PATH", help="Write trades to CSV.")
        if name in ("compare", "report"):
            s.add_argument(
                "--sma", type=int, default=200,
                help="Moving average period for the trend baseline (default 200).",
            )
        if name == "report":
            s.add_argument("--out", default="report.html", metavar="PATH",
                           help="Where to write the page (default report.html).")
            s.add_argument("--title", default="BTC Strategy Report")
        if name in ("scan", "demo"):
            s.add_argument(
                "--show-levels", type=int, default=10, help="Level rows to print (default 10)."
            )
    return p


def _add_data_args(s: argparse.ArgumentParser) -> None:
    g = s.add_argument_group("data")
    g.add_argument("--trades", metavar="PATH", help="Tick CSV: ts, price, size, side.")
    g.add_argument("--bars", metavar="PATH", help="OHLCV CSV (proxy footprints unless it carries ask/bid volume).")
    g.add_argument(
        "--binance-klines",
        nargs="+",
        metavar="PATH",
        help="Binance monthly kline .zip/.csv files or a directory of them. "
        "Carries measured taker-buy volume, so bar delta is real.",
    )
    g.add_argument(
        "--binance-aggtrades",
        nargs="+",
        metavar="PATH",
        help="Binance monthly aggTrades .zip/.csv files or a directory. "
        "The real tape, giving true footprints.",
    )
    g.add_argument(
        "--funding",
        nargs="+",
        metavar="PATH",
        help="Binance fundingRate .zip/.csv files or a directory. Charges perp "
        "funding against open positions. Omit for spot.",
    )
    g.add_argument(
        "--timeframe",
        type=float,
        default=300.0,
        help="Bar length in seconds when aggregating trades (default 300).",
    )
    g.add_argument(
        "--tick-size",
        type=float,
        help="Footprint row height (NOT the exchange tick). Aim for 20-50 rows per bar.",
    )
    g.add_argument("--tick-value", type=float, help="Currency value of one row.")
    g.add_argument("--synthetic", action="store_true", help="Use a generated tape.")
    g.add_argument("--seed", type=int, default=7, help="Synthetic tape seed.")
    g.add_argument("--ticks", type=int, default=240_000, help="Synthetic tape length in prints.")


def _add_strategy_args(s: argparse.ArgumentParser) -> None:
    g = s.add_argument_group("strategy")
    g.add_argument(
        "--preset",
        choices=sorted(PRESETS),
        help="Instrument defaults (row size, fee model, sizing). Explicit flags win.",
    )
    g.add_argument("--entry-mode", choices=ENTRY_MODES, help="Where the entry goes on candle 2.")
    g.add_argument("--regime", choices=REGIME_FILTERS, dest="regime_filter", help="Trend filter.")
    g.add_argument("--target-r", type=float, help="Target as a multiple of risk.")
    g.add_argument("--stop-buffer-atr", type=float, help="Stop padding beyond the level, in ATR.")
    g.add_argument("--partial-at-r", type=float, help="Scale-out point in R (0 disables).")
    g.add_argument("--trail-atr", type=float, help="Trailing stop distance in ATR (0 disables).")
    g.add_argument("--time-stop", type=int, dest="time_stop_bars", help="Flatten after N bars.")
    g.add_argument("--risk", type=float, dest="risk_per_trade", help="Fraction of equity per trade.")
    g.add_argument("--equity", type=float, dest="starting_equity", help="Starting equity.")
    g.add_argument("--slippage-ticks", type=float, help="Slippage per market/stop fill, in rows.")
    g.add_argument(
        "--slippage-pct", type=float, help="Slippage as a fraction of price (0.0002 = 2 bp)."
    )
    g.add_argument("--commission", type=float, dest="commission_per_side", help="Per unit per side.")
    g.add_argument(
        "--commission-pct",
        type=float,
        help="Commission as a fraction of notional per side (0.001 = 10 bp).",
    )
    g.add_argument("--wick-frac", type=float, help="Minimum candle-1 rejection wick fraction.")
    g.add_argument(
        "--c2-close-frac",
        type=float,
        dest="c2_close_beyond_c1_frac",
        help="How far through candle 1 candle 2 must close (0-1).",
    )
    g.add_argument(
        "--session-start",
        type=int,
        dest="session_start_min",
        help="Session open, minutes from UTC midnight (e.g. 810 for 13:30 UTC).",
    )
    g.add_argument(
        "--session-end",
        type=int,
        dest="session_end_min",
        help="Session close, minutes from UTC midnight. Wraps past midnight if below start.",
    )
    g.add_argument("--no-longs", action="store_true", help="Only trade shorts at resistance.")
    g.add_argument("--no-shorts", action="store_true", help="Only trade longs at support.")
    g.add_argument(
        "--allow-loose-flow",
        action="store_true",
        help="Accept a rejection wick without order flow confirmation (not recommended).",
    )
    g.add_argument(
        "--allow-chart-levels",
        action="store_true",
        help="Also trade levels with no order-flow signature (wicks and volume nodes only).",
    )


def config_from_args(args: argparse.Namespace) -> StrategyConfig:
    overrides = {}
    simple = (
        "entry_mode",
        "regime_filter",
        "target_r",
        "stop_buffer_atr",
        "risk_per_trade",
        "starting_equity",
        "slippage_ticks",
        "slippage_pct",
        "commission_per_side",
        "commission_pct",
        "wick_frac",
        "c2_close_beyond_c1_frac",
        "time_stop_bars",
        "session_start_min",
        "session_end_min",
        "tick_size",
        "tick_value",
    )
    for key in simple:
        value = getattr(args, key, None)
        if value is not None:
            overrides[key] = value

    if getattr(args, "partial_at_r", None) is not None:
        overrides["partial_at_r"] = args.partial_at_r if args.partial_at_r > 0 else None
    if getattr(args, "trail_atr", None) is not None:
        overrides["trail_atr"] = args.trail_atr if args.trail_atr > 0 else None
    if getattr(args, "no_longs", False):
        overrides["trade_longs"] = False
    if getattr(args, "no_shorts", False):
        overrides["trade_shorts"] = False
    if getattr(args, "allow_loose_flow", False):
        overrides["require_flow_evidence"] = False
    if getattr(args, "allow_chart_levels", False):
        overrides["require_flow_backed_levels"] = False

    preset = getattr(args, "preset", None)
    if preset:
        # Preset supplies the instrument's conventions; explicit flags win.
        return build_preset(preset, **overrides)
    return StrategyConfig(**overrides)


def load_funding(args: argparse.Namespace, cfg: StrategyConfig) -> Optional[FundingSchedule]:
    paths = getattr(args, "funding", None)
    if not paths:
        return None
    sched = load_binance_paths(paths, cfg.tick_size, kind="fundingRate")
    if sched.is_empty:
        raise SystemExit("no funding settlements found in the given paths")
    print(f"Funding: {sched.summary()}\n")
    return sched


def load_bars(args: argparse.Namespace, cfg: StrategyConfig) -> List[Bar]:
    tick = cfg.tick_size
    if args.trades:
        trades = load_trades_csv(args.trades)
        if not trades:
            raise SystemExit(f"no trades found in {args.trades}")
        return bars_from_trades(trades, args.timeframe, tick)
    if args.bars:
        bars = load_bars_csv(args.bars, tick)
        if not bars:
            raise SystemExit(f"no bars found in {args.bars}")
        return bars
    if getattr(args, "binance_klines", None):
        bars = load_binance_paths(args.binance_klines, tick, kind="klines")
        if not bars:
            raise SystemExit("no bars found in the Binance kline archives")
        return bars
    if getattr(args, "binance_aggtrades", None):
        trades = load_binance_paths(args.binance_aggtrades, tick, kind="aggTrades")
        if not trades:
            raise SystemExit("no trades found in the Binance aggTrades archives")
        return bars_from_trades(trades, args.timeframe, tick)
    if args.synthetic or args.command == "demo":
        sc = SyntheticConfig(n_ticks=args.ticks, tick_size=tick, seed=args.seed)
        return bars_from_trades(synthetic_trades(sc), args.timeframe, tick)
    raise SystemExit(
        "provide a data source: --trades, --bars, --binance-klines, "
        "--binance-aggtrades, or --synthetic"
    )


def export_trades(path: str, trades) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(
            [
                "direction", "entry_ts", "entry_price", "exit_ts", "exit_price",
                "quantity", "pnl", "r_multiple", "mae_r", "mfe_r", "bars_held",
                "exit_reason", "level_price", "level_strength", "reasons",
            ]
        )
        for t in trades:
            w.writerow(
                [
                    "short" if t.direction < 0 else "long",
                    f"{t.entry_ts:.0f}", f"{t.entry_price:.4f}",
                    f"{t.exit_ts:.0f}", f"{t.exit_price:.4f}",
                    f"{t.quantity:.4f}", f"{t.pnl:.2f}", f"{t.r_multiple:.4f}",
                    f"{t.mae_r:.3f}", f"{t.mfe_r:.3f}", t.bars_held,
                    t.exit_reason, f"{t.level_price:.4f}", f"{t.level_strength:.2f}",
                    " | ".join(t.reasons),
                ]
            )


def cmd_backtest(args: argparse.Namespace, bars: List[Bar], cfg: StrategyConfig) -> None:
    result = Backtester(cfg).run(bars, funding=load_funding(args, cfg))
    stats = summarize(result)
    print(format_stats(stats, title=f"Backtest over {len(bars)} bars"))
    limit = None if args.show_trades == 0 else args.show_trades
    print(format_trades(result.trades, limit))
    if args.export_trades:
        export_trades(args.export_trades, result.trades)
        print(f"Wrote {len(result.trades)} trades to {args.export_trades}\n")


def cmd_optimize(args: argparse.Namespace, bars: List[Bar], cfg: StrategyConfig) -> None:
    grid = FAST_GRID if args.fast else DEFAULT_GRID
    combos = 1
    for values in grid.values():
        combos *= len(values)
    print(f"Searching {combos} configurations over {args.folds} folds ({len(bars)} bars)...\n")
    rec = walk_forward(bars, cfg, grid=grid, folds=args.folds, min_trades=args.min_trades)
    print(format_recommendation(rec, cfg))
    print(DISCLAIMER)


def cmd_scan(args: argparse.Namespace, bars: List[Bar], cfg: StrategyConfig) -> None:
    """Replay the data and report the current state, as if it were live."""
    ind = prepare(bars, cfg)
    engine = SignalEngine(cfg)
    last_signals = []
    for i in range(len(bars)):
        sigs = engine.on_bar(i, bars, ind)
        if sigs:
            last_signals = [(i, s) for s in sigs]

    idx = len(bars) - 1
    print(format_levels(engine.book.snapshot(idx), idx, cfg, limit=args.show_levels))

    armed = [s for s in engine.setups if s.expires_at >= idx]
    if armed:
        print("Armed candle-1 rejections waiting on confirmation:")
        for s in armed:
            side = "short" if s.direction < 0 else "long"
            print(f"  {side} at {s.level.price:.2f} (bar {s.c1_index}): {'; '.join(s.reasons)}")
        print()

    if last_signals and last_signals[-1][0] == idx:
        for _, sig in last_signals:
            print(format_signal(sig, cfg))
    elif last_signals:
        i, sig = last_signals[-1]
        print(f"No setup on the latest bar. Most recent was {idx - i} bars ago:\n")
        print(format_signal(sig, cfg))
    else:
        print("No confirmed setups in this data.\n")
    print(DISCLAIMER)


def cmd_demo(args: argparse.Namespace, bars: List[Bar], cfg: StrategyConfig) -> None:
    print(
        f"Synthetic tape: {len(bars)} bars of {args.timeframe:.0f}s "
        f"(seed {args.seed}).\n"
        "The generator plants real passive blocks that price trades into, so\n"
        "absorption genuinely exists in this data. It has no news, no other\n"
        "participants and no reflexivity, so treat what follows as proof the\n"
        "code works -- never as evidence of an edge.\n"
    )
    result = Backtester(cfg).run(bars)
    stats = summarize(result)
    print(format_stats(stats, title=f"Demo backtest over {len(bars)} bars"))
    limit = None if args.show_trades == 0 else args.show_trades
    print(format_trades(result.trades, limit))
    idx = len(bars) - 1
    engine = SignalEngine(cfg)
    ind = prepare(bars, cfg)
    for i in range(len(bars)):
        engine.on_bar(i, bars, ind)
    print(format_levels(engine.book.snapshot(idx), idx, cfg, limit=args.show_levels))
    if args.export_trades:
        export_trades(args.export_trades, result.trades)
        print(f"Wrote {len(result.trades)} trades to {args.export_trades}\n")


def cmd_compare(args: argparse.Namespace, bars: List[Bar], cfg: StrategyConfig) -> None:
    """The order-flow strategy next to the two things it has to beat."""
    funding = load_funding(args, cfg)
    if len(bars) <= args.sma + 10:
        print(
            f"WARNING: {len(bars)} bars against an SMA{args.sma} leaves "
            f"{max(0, len(bars)-args.sma)} tradeable -- expect a handful of trades\n"
            "and no statistical meaning. Use a longer history for the trend leg.\n"
        )

    rows = []
    of_result = Backtester(cfg).run(bars, funding=funding)
    rows.append(("order flow (2-candle)", summarize(of_result)))
    if len(bars) > args.sma + 2:
        rows.append((f"trend SMA{args.sma}", summarize(trend_baseline(bars, cfg, args.sma, funding=funding))))
    rows.append(("buy and hold", summarize(buy_and_hold(bars, cfg))))

    print(format_comparison(rows))
    print(format_confluence(analyse_confluence(of_result.trades)))
    for name, st in rows:
        print(format_stats(st, title=name))


def _run_all(args, bars, cfg):
    """The three strategies plus the confluence analysis, in one place."""
    funding = load_funding(args, cfg)
    of_result = Backtester(cfg).run(bars, funding=funding)
    sections = [("Order flow", of_result, summarize(of_result))]
    if len(bars) > args.sma + 2:
        tr = trend_baseline(bars, cfg, args.sma, funding=funding)
        sections.append((f"Trend SMA{args.sma}", tr, summarize(tr)))
    bh = buy_and_hold(bars, cfg)
    sections.append(("Buy and hold", bh, summarize(bh)))
    return sections, analyse_confluence(of_result.trades)


def cmd_report(args: argparse.Namespace, bars: List[Bar], cfg: StrategyConfig) -> None:
    sections, conf = _run_all(args, bars, cfg)
    note = "synthetic data" if getattr(args, "synthetic", False) else ""
    page = build_report(sections, cfg, confluence=conf, title=args.title, data_note=note)
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write(page)
    print(format_confluence(conf))
    print(f"Wrote {len(page):,} bytes to {args.out}")
    print("Open it in a browser, or publish it wherever your team reads things.")


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = config_from_args(args)
    bars = load_bars(args, cfg)

    if len(bars) < cfg.atr_period + cfg.swing_lookback + 5:
        raise SystemExit(f"not enough bars ({len(bars)}) to run the strategy")

    if args.command == "backtest":
        cmd_backtest(args, bars, cfg)
    elif args.command == "optimize":
        cmd_optimize(args, bars, cfg)
    elif args.command == "scan":
        cmd_scan(args, bars, cfg)
    elif args.command == "demo":
        cmd_demo(args, bars, cfg)
    elif args.command == "compare":
        cmd_compare(args, bars, cfg)
    elif args.command == "report":
        cmd_report(args, bars, cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
