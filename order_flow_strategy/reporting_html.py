"""Renders a run into a single self-contained HTML page.

Terminal tables are fine for one number and hopeless for comparing three
strategies across eight measures. This produces one page you can open, keep,
or send to someone: the verdict first, the equity curves, the confluence
analysis, then the detail.

No external dependencies and no network at render time -- the CSS, the charts
(hand-built inline SVG) and the data all ship inside the file, so it opens from
disk on a machine with no internet. The only remote request is the webfont,
which degrades to the declared fallback stack.
"""

import html
import time
from typing import Dict, List, Optional, Sequence, Tuple

from .backtest import BacktestResult
from .config import StrategyConfig
from .confluence import ConfluenceReport, FactorStats
from .metrics import PerformanceStats
from .report import warnings_for

Section = Tuple[str, BacktestResult, PerformanceStats]

#: Series colours, chosen to stay distinguishable on both grounds.
SERIES_COLOURS = ("#2E7A80", "#B07C3A", "#6B7A8F")


# ----------------------------------------------------------------------
# SVG helpers
# ----------------------------------------------------------------------
def _svg_equity(sections: Sequence[Section], width: int = 720, height: int = 260) -> str:
    """Equity curves, each rebased to 100 so shapes compare directly."""
    series = [(name, res.equity_curve) for name, res, _ in sections if res.equity_curve]
    if not series:
        return '<p class="empty">No equity curve to draw.</p>'

    pad_l, pad_r, pad_t, pad_b = 52, 12, 14, 28
    plot_w = width - pad_l - pad_r
    plot_h = height - pad_t - pad_b

    rebased = []
    for name, curve in series:
        base = curve[0] if curve[0] else 1.0
        rebased.append((name, [100.0 * v / base for v in curve]))

    lo = min(min(vals) for _, vals in rebased)
    hi = max(max(vals) for _, vals in rebased)
    if hi - lo < 1e-9:
        lo, hi = lo - 1, hi + 1
    span = hi - lo

    def y(v: float) -> float:
        return pad_t + plot_h * (1 - (v - lo) / span)

    parts = [
        f'<svg viewBox="0 0 {width} {height}" role="img" '
        f'aria-label="Equity curves rebased to 100" class="chart">'
    ]

    # Gridlines and value labels.
    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        v = lo + span * frac
        yy = y(v)
        parts.append(
            f'<line x1="{pad_l}" y1="{yy:.1f}" x2="{width-pad_r}" y2="{yy:.1f}" '
            f'class="grid" />'
        )
        parts.append(
            f'<text x="{pad_l-8}" y="{yy+4:.1f}" text-anchor="end" '
            f'class="tick">{v:.0f}</text>'
        )

    # The 100 line is the "made nothing" reference, so mark it properly.
    if lo <= 100.0 <= hi:
        parts.append(
            f'<line x1="{pad_l}" y1="{y(100):.1f}" x2="{width-pad_r}" '
            f'y2="{y(100):.1f}" class="baseline" />'
        )

    for i, (name, vals) in enumerate(rebased):
        colour = SERIES_COLOURS[i % len(SERIES_COLOURS)]
        n = len(vals)
        step = plot_w / max(n - 1, 1)
        pts = " ".join(f"{pad_l + j*step:.2f},{y(v):.2f}" for j, v in enumerate(vals))
        parts.append(
            f'<polyline points="{pts}" fill="none" stroke="{colour}" '
            f'stroke-width="1.8" stroke-linejoin="round" />'
        )
        # Emphasise where each series ends -- that is the number people want.
        parts.append(
            f'<circle cx="{pad_l + (n-1)*step:.2f}" cy="{y(vals[-1]):.2f}" r="3.2" '
            f'fill="{colour}" />'
        )

    parts.append(
        f'<text x="{pad_l}" y="{height-8}" class="tick">start</text>'
        f'<text x="{width-pad_r}" y="{height-8}" text-anchor="end" class="tick">end</text>'
    )
    parts.append("</svg>")
    return "".join(parts)


def _svg_factor_bars(factors: Sequence[FactorStats], width: int = 720) -> str:
    """Diverging bars: expectancy lift in R for each condition."""
    if not factors:
        return '<p class="empty">No tagged trades, so nothing to attribute.</p>'

    row_h, pad_t, pad_b = 30, 10, 26
    label_w, pad_r = 150, 60
    height = pad_t + pad_b + row_h * len(factors)
    plot_w = width - label_w - pad_r
    mid = label_w + plot_w / 2

    reach = max(0.25, max(abs(f.lift) for f in factors))

    parts = [
        f'<svg viewBox="0 0 {width} {height}" role="img" '
        f'aria-label="Expectancy lift by condition" class="chart">'
    ]
    parts.append(
        f'<line x1="{mid}" y1="{pad_t}" x2="{mid}" y2="{height-pad_b}" class="baseline" />'
    )

    for i, f in enumerate(factors):
        cy = pad_t + row_h * i + row_h / 2
        w = (abs(f.lift) / reach) * (plot_w / 2 - 8)
        x = mid if f.lift >= 0 else mid - w
        cls = "bar-gain" if f.lift >= 0 else "bar-loss"
        if f.verdict == "too few to judge":
            cls = "bar-weak"
        parts.append(
            f'<rect x="{x:.1f}" y="{cy-8:.1f}" width="{max(w,1):.1f}" height="16" '
            f'rx="2" class="{cls}" />'
        )
        parts.append(
            f'<text x="{label_w-10}" y="{cy+4:.1f}" text-anchor="end" '
            f'class="rowlab">{html.escape(f.tag)}</text>'
        )
        parts.append(
            f'<text x="{width-pad_r+8}" y="{cy+4:.1f}" class="rowval">'
            f'{f.lift:+.2f}R</text>'
        )

    parts.append(
        f'<text x="{mid}" y="{height-8}" text-anchor="middle" class="tick">'
        f'expectancy with the condition minus without</text>'
    )
    parts.append("</svg>")
    return "".join(parts)


# ----------------------------------------------------------------------
# Fragments
# ----------------------------------------------------------------------
def _verdict(sections: Sequence[Section]) -> Tuple[str, str]:
    """Headline sentence plus a tone class. Deliberately blunt."""
    by_name = {name: st for name, _, st in sections}
    hold = next((st for n, st in by_name.items() if "hold" in n.lower()), None)
    others = [(n, st) for n, st in by_name.items() if "hold" not in n.lower()]
    if not hold or not others:
        return ("Not enough strategies to compare.", "flat")

    best_name, best = max(others, key=lambda kv: kv[1].return_pct)
    if best.n_trades == 0:
        return ("No strategy produced a trade on this data.", "flat")
    if best.return_pct > hold.return_pct:
        return (
            f"{best_name} returned {best.return_pct*100:.1f}% against "
            f"{hold.return_pct*100:.1f}% for buying and holding — but check the "
            f"t-statistic ({best.t_stat:+.1f}) before believing it.",
            "gain" if best.t_stat >= 2 else "flat",
        )
    return (
        f"Nothing beat buying and holding. Best was {best_name} at "
        f"{best.return_pct*100:.1f}% against {hold.return_pct*100:.1f}%.",
        "loss",
    )


def _stat_row(label: str, value: str, note: str = "") -> str:
    note_html = f'<span class="note">{html.escape(note)}</span>' if note else ""
    return (
        f'<div class="stat"><dt>{html.escape(label)}</dt>'
        f'<dd>{value}{note_html}</dd></div>'
    )


def _comparison_table(sections: Sequence[Section]) -> str:
    head = (
        "<tr><th>Strategy</th><th>Trades</th><th>Win</th><th>Expectancy</th>"
        "<th>t</th><th>Return</th><th>Max DD</th><th>Cost</th></tr>"
    )
    rows = []
    for name, _, st in sections:
        if st.n_trades == 0:
            rows.append(
                f'<tr><td>{html.escape(name)}</td>'
                f'<td colspan="7" class="muted">no trades</td></tr>'
            )
            continue
        tone = "gain" if st.return_pct > 0 else "loss"
        rows.append(
            f"<tr><td>{html.escape(name)}</td>"
            f"<td>{st.n_trades}</td>"
            f"<td>{st.win_rate*100:.1f}%</td>"
            f'<td class="{"gain" if st.expectancy_r > 0 else "loss"}">'
            f"{st.expectancy_r:+.2f}R</td>"
            f"<td>{st.t_stat:+.1f}</td>"
            f'<td class="{tone}">{st.return_pct*100:+.1f}%</td>'
            f"<td>{st.max_drawdown_pct*100:.1f}%</td>"
            f"<td>{st.avg_cost_r:.2f}R</td></tr>"
        )
    return (
        '<div class="scroll"><table class="grid-table">'
        f"<thead>{head}</thead><tbody>{''.join(rows)}</tbody></table></div>"
    )


def _confluence_section(rep: ConfluenceReport) -> str:
    if not rep.factors:
        return ""
    bars = _svg_factor_bars(rep.factors)

    chips = []
    for f in rep.factors:
        cls = {
            "earns its place": "chip-gain",
            "hurts -- consider dropping": "chip-loss",
            "no measurable effect": "chip-flat",
        }.get(f.verdict, "chip-weak")
        chips.append(
            f'<li class="factor"><span class="factor-name">{html.escape(f.tag)}</span>'
            f'<span class="{cls}">{html.escape(f.verdict)}</span>'
            f'<span class="factor-meta">present in {f.coverage*100:.0f}% of trades · '
            f'{f.exp_with:+.2f}R with · {f.exp_without:+.2f}R without</span></li>'
        )

    counts = []
    for c, cs in sorted(rep.by_count.items()):
        counts.append(
            f"<tr><td>{c}</td><td>{cs.n}</td>"
            f'<td class="{"gain" if cs.expectancy > 0 else "loss"}">'
            f"{cs.expectancy:+.2f}R</td><td>{cs.win_rate*100:.0f}%</td></tr>"
        )

    stack_verdict = (
        "Expectancy rose with the number of conditions present."
        if rep.more_is_better
        else "Expectancy did <strong>not</strong> rise with more conditions present."
    )

    return f"""
    <section id="confluence">
      <h2>Confluence</h2>
      <p class="lede">The strategy stacks evidence before it fires. The usual
      assumption is that more agreement means a better trade. That is testable,
      so here it is tested: each condition split into the trades that had it and
      the trades that did not.</p>
      {bars}
      <ul class="factors">{''.join(chips)}</ul>
      <div class="split">
        <div>
          <h3>Does stacking help?</h3>
          <div class="scroll"><table class="grid-table">
            <thead><tr><th>Conditions</th><th>Trades</th><th>Expectancy</th><th>Win</th></tr></thead>
            <tbody>{''.join(counts)}</tbody></table></div>
          <p class="note-block">{stack_verdict}</p>
        </div>
        <div>
          <h3>Read this carefully</h3>
          <p>These are subgroups of a single sample, not independent experiments.
          Split six ways and one will look good by chance alone. A lift here is a
          hypothesis to test on fresh data — it is not a filter to switch on
          today, and acting on it is how a backtest quietly becomes a curve fit.</p>
        </div>
      </div>
    </section>"""


def _warnings_block(sections: Sequence[Section]) -> str:
    items = []
    for name, _, st in sections:
        for w in warnings_for(st):
            items.append(f"<li><strong>{html.escape(name)}</strong> — {html.escape(w)}</li>")
    if not items:
        return ""
    return (
        '<section id="caveats"><h2>Before you believe any of this</h2>'
        f'<ul class="warnings">{"".join(items)}</ul></section>'
    )


def _detail(sections: Sequence[Section]) -> str:
    blocks = []
    for name, _, st in sections:
        if st.n_trades == 0:
            continue
        rows = [
            _stat_row("Trades", f"{st.n_trades}", f"{st.n_long} long / {st.n_short} short"),
            _stat_row("Win rate", f"{st.win_rate*100:.1f}%", f"{st.n_wins}W / {st.n_losses}L"),
            _stat_row("Expectancy", f"{st.expectancy_r:+.2f}R", f"t = {st.t_stat:+.2f}"),
            _stat_row("Payoff", f"{st.payoff_ratio:.2f}",
                      f"{st.avg_win_r:+.2f}R / {st.avg_loss_r:+.2f}R"),
            _stat_row("Profit factor", f"{st.profit_factor:.2f}"),
            _stat_row("Cost hurdle", f"{st.avg_cost_r:.2f}R",
                      "fees and slippage per trade"),
            _stat_row("Max drawdown", f"{st.max_drawdown_pct*100:.1f}%",
                      f"worst streak {st.max_consecutive_losses}"),
            _stat_row("Heat on winners", f"{st.avg_mae_r_wins:.2f}R",
                      "stop too tight above ~0.8"),
            _stat_row("Run-up on losers", f"{st.avg_mfe_r_losses:.2f}R",
                      "target too far above ~1.0"),
        ]
        if st.total_funding:
            rows.append(_stat_row("Perp funding", f"{st.total_funding:,.0f}",
                                  f"{st.avg_funding_r:+.2f}R per trade"))
        exits = ", ".join(f"{k} {v}" for k, v in sorted(st.exit_reasons.items()))
        blocks.append(
            f'<article class="detail"><h3>{html.escape(name)}</h3>'
            f'<dl class="stats">{"".join(rows)}</dl>'
            f'<p class="exits"><span class="lab">Exits</span> {html.escape(exits)}</p></article>'
        )
    return f'<section id="detail"><h2>By strategy</h2>{"".join(blocks)}</section>'


# ----------------------------------------------------------------------
def build_report(
    sections: Sequence[Section],
    cfg: StrategyConfig,
    confluence: Optional[ConfluenceReport] = None,
    title: str = "BTC Strategy Report",
    data_note: str = "",
) -> str:
    """Render the whole page. Returns HTML as a string."""
    verdict, tone = _verdict(sections)
    generated = time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime())

    spans = [res for _, res, _ in sections if res.bar_timestamps]
    if spans:
        ts = spans[0].bar_timestamps
        days = (ts[-1] - ts[0]) / 86400.0
        span_txt = f"{len(ts):,} bars · {days:.0f} days"
    else:
        span_txt = "no bars"

    legend = "".join(
        f'<span class="key"><i style="background:{SERIES_COLOURS[i % 3]}"></i>'
        f"{html.escape(name)}</span>"
        for i, (name, _, _) in enumerate(sections)
    )

    setup = " · ".join(
        [
            f"row size ${cfg.tick_size:g}",
            f"risk {cfg.risk_per_trade*100:g}% per trade",
            f"target {cfg.target_r:g}R",
            f"fees {cfg.commission_pct*1e4:g}bp",
            f"slippage {cfg.slippage_pct*1e4:g}bp",
            f"leverage cap {cfg.max_leverage if cfg.max_leverage else 'none'}",
        ]
    )

    return f"""<title>{html.escape(title)}</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Serif:wght@500;600&display=swap">
<style>
:root {{
  --ground:#F3F5F4; --surface:#FFFFFF; --sunken:#EAEEEC;
  --ink:#1A1F21; --ink-2:#4A5457; --ink-3:#767F82;
  --line:#D7DEDB; --accent:#2E7A80; --gain:#37705A; --loss:#A2504A;
  --gain-soft:#37705A2E; --loss-soft:#A2504A2E; --weak:#9AA5A8;
}}
@media (prefers-color-scheme: dark) {{
  :root:not([data-theme="light"]) {{
    --ground:#13181A; --surface:#1A2124; --sunken:#101517;
    --ink:#E6EBEA; --ink-2:#A7B3B3; --ink-3:#7A8688;
    --line:#2A3438; --accent:#5FB3B8; --gain:#5FA083; --loss:#C97A70;
    --gain-soft:#5FA0833D; --loss-soft:#C97A703D; --weak:#5D686B;
  }}
}}
:root[data-theme="dark"] {{
  --ground:#13181A; --surface:#1A2124; --sunken:#101517;
  --ink:#E6EBEA; --ink-2:#A7B3B3; --ink-3:#7A8688;
  --line:#2A3438; --accent:#5FB3B8; --gain:#5FA083; --loss:#C97A70;
  --gain-soft:#5FA0833D; --loss-soft:#C97A703D; --weak:#5D686B;
}}
*{{box-sizing:border-box}}
body{{background:var(--ground);color:var(--ink);
  font:16px/1.6 "IBM Plex Sans",system-ui,-apple-system,sans-serif;
  margin:0;padding:0 20px 72px}}
.wrap{{max-width:820px;margin:0 auto}}
h1,h2,h3{{font-family:"IBM Plex Serif",Georgia,serif;text-wrap:balance;
  letter-spacing:-.01em;margin:0}}
h1{{font-size:2rem;font-weight:600;line-height:1.2}}
h2{{font-size:1.3rem;font-weight:600;margin:44px 0 6px;
  padding-bottom:8px;border-bottom:1px solid var(--line)}}
h3{{font-size:1rem;font-weight:600;margin:22px 0 8px}}
p{{margin:.6em 0;max-width:66ch}}
.eyebrow{{font-size:.72rem;letter-spacing:.14em;text-transform:uppercase;
  color:var(--accent);font-weight:600;margin:0 0 10px}}
header{{padding:44px 0 8px}}
.meta{{font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:.78rem;
  color:var(--ink-3);margin-top:10px}}
.verdict{{margin:22px 0 8px;padding:16px 18px;border-radius:3px;
  border-left:3px solid var(--weak);background:var(--surface);
  font-size:1.02rem;line-height:1.5}}
.verdict.gain{{border-left-color:var(--gain);background:var(--gain-soft)}}
.verdict.loss{{border-left-color:var(--loss);background:var(--loss-soft)}}
.setup{{font-family:"IBM Plex Mono",monospace;font-size:.74rem;color:var(--ink-3);
  background:var(--sunken);padding:8px 12px;border-radius:3px;margin-top:14px;
  overflow-x:auto;white-space:nowrap}}
.lede{{color:var(--ink-2)}}
.chart{{width:100%;height:auto;display:block;margin:14px 0 6px;color:var(--ink-3)}}
.grid{{stroke:var(--line);stroke-width:1}}
.baseline{{stroke:var(--ink-3);stroke-width:1;stroke-dasharray:3 3}}
.tick{{fill:currentColor;font-family:"IBM Plex Mono",monospace;font-size:10px}}
.rowlab{{fill:var(--ink);font-family:"IBM Plex Sans",sans-serif;font-size:12px}}
.rowval{{fill:var(--ink-2);font-family:"IBM Plex Mono",monospace;font-size:11px}}
.bar-gain{{fill:var(--gain)}} .bar-loss{{fill:var(--loss)}} .bar-weak{{fill:var(--weak)}}
.legend{{display:flex;gap:18px;flex-wrap:wrap;font-size:.8rem;color:var(--ink-2)}}
.key{{display:flex;align-items:center;gap:7px}}
.key i{{width:14px;height:3px;border-radius:2px;display:inline-block}}
.scroll{{overflow-x:auto;margin:12px 0}}
table.grid-table{{width:100%;border-collapse:collapse;
  font-family:"IBM Plex Mono",monospace;font-size:.8rem;
  font-variant-numeric:tabular-nums}}
.grid-table th{{text-align:right;font-weight:500;color:var(--ink-3);
  padding:6px 10px;border-bottom:1px solid var(--line);white-space:nowrap;
  font-family:"IBM Plex Sans",sans-serif;font-size:.72rem;
  letter-spacing:.05em;text-transform:uppercase}}
.grid-table td{{text-align:right;padding:7px 10px;
  border-bottom:1px solid var(--line);white-space:nowrap}}
.grid-table th:first-child,.grid-table td:first-child{{text-align:left}}
.gain{{color:var(--gain)}} .loss{{color:var(--loss)}} .muted{{color:var(--ink-3)}}
.factors{{list-style:none;padding:0;margin:18px 0;display:grid;gap:1px;
  background:var(--line);border:1px solid var(--line);border-radius:3px;
  overflow:hidden}}
.factor{{background:var(--surface);padding:11px 14px;display:grid;
  grid-template-columns:1fr auto;gap:4px 12px;align-items:baseline}}
.factor-name{{font-family:"IBM Plex Mono",monospace;font-size:.84rem}}
.factor-meta{{grid-column:1/-1;font-size:.74rem;color:var(--ink-3);
  font-family:"IBM Plex Mono",monospace;font-variant-numeric:tabular-nums}}
.chip-gain,.chip-loss,.chip-flat,.chip-weak{{font-size:.68rem;padding:2px 8px;
  border-radius:2px;letter-spacing:.04em;text-transform:uppercase;font-weight:600;
  white-space:nowrap}}
.chip-gain{{background:var(--gain-soft);color:var(--gain)}}
.chip-loss{{background:var(--loss-soft);color:var(--loss)}}
.chip-flat{{background:var(--sunken);color:var(--ink-3)}}
.chip-weak{{background:var(--sunken);color:var(--weak)}}
.split{{display:grid;grid-template-columns:1fr 1fr;gap:28px;margin-top:20px}}
@media (max-width:640px){{.split{{grid-template-columns:1fr}}}}
.note-block{{font-size:.85rem;color:var(--ink-2)}}
.stats{{display:grid;grid-template-columns:repeat(auto-fit,minmax(168px,1fr));
  gap:1px;background:var(--line);border:1px solid var(--line);
  border-radius:3px;overflow:hidden;margin:10px 0}}
.stat{{background:var(--surface);padding:10px 13px}}
.stat dt{{font-size:.68rem;letter-spacing:.07em;text-transform:uppercase;
  color:var(--ink-3);margin-bottom:3px}}
.stat dd{{margin:0;font-family:"IBM Plex Mono",monospace;font-size:1.02rem;
  font-variant-numeric:tabular-nums}}
.stat .note{{display:block;font-size:.7rem;color:var(--ink-3);margin-top:2px;
  font-family:"IBM Plex Sans",sans-serif}}
.detail{{margin-bottom:26px}}
.exits{{font-family:"IBM Plex Mono",monospace;font-size:.76rem;color:var(--ink-3)}}
.exits .lab{{text-transform:uppercase;letter-spacing:.07em;font-size:.66rem;
  margin-right:8px}}
.warnings{{margin:12px 0;padding-left:20px}}
.warnings li{{margin:8px 0;color:var(--ink-2);max-width:66ch}}
.warnings strong{{color:var(--ink);font-weight:600}}
footer{{margin-top:52px;padding-top:18px;border-top:1px solid var(--line);
  font-size:.8rem;color:var(--ink-3);max-width:66ch}}
.empty{{color:var(--ink-3);font-style:italic}}
</style>

<div class="wrap">
<header>
  <p class="eyebrow">Backtest</p>
  <h1>{html.escape(title)}</h1>
  <p class="meta">{html.escape(span_txt)} · generated {generated}
    {(' · ' + html.escape(data_note)) if data_note else ''}</p>
  <div class="verdict {tone}">{verdict}</div>
  <div class="setup">{html.escape(setup)}</div>
</header>

<section id="curves">
  <h2>Equity</h2>
  <p class="lede">Each series rebased to 100 at the first bar, so the shapes
  compare regardless of stake. The dashed line is break-even.</p>
  {_svg_equity(sections)}
  <div class="legend">{legend}</div>
  {_comparison_table(sections)}
</section>

{_confluence_section(confluence) if confluence else ''}
{_detail(sections)}
{_warnings_block(sections)}

<footer>
  Research output, not financial advice and not a prediction. Live fills,
  queue position and data quality will all be worse than modelled here.
  Backtested results describe what already happened to one sample.
</footer>
</div>
"""
