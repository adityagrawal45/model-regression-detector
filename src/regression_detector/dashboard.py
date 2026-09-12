"""Renders a standalone HTML trend dashboard from stored eval history.

Where report_html.py shows one diff (baseline vs candidate, last 20 runs of
context), this shows the whole picture: accuracy/latency/token trends across
every run in history_store.py's SQLite index, current drift status, and a
table of recent runs linking back to each run's own JSON report.

Same self-contained-file conventions as report_html.py: inline CSS, inline
SVG charts, no external assets, no JS.

Usage:
    python -m regression_detector.dashboard \\
        --history-db reports/history.db --out reports/dashboard.html
"""

from __future__ import annotations

import argparse
import html
from pathlib import Path
from typing import TYPE_CHECKING

from regression_detector.eval_runner import EvalReport
from regression_detector.history_store import DEFAULT_DB_PATH, load_history, load_recent_run_rows

if TYPE_CHECKING:  # avoid a circular import - drift.py is imported lazily in main() below
    from regression_detector.drift import DriftReport

DEFAULT_TREND_POINTS = 50
DEFAULT_RECENT_ROWS = 20

_SEVERITY_COLOR = {"none": "#2e7d32", "warning": "#b26a00", "critical": "#c62828"}


def _esc(value: str | None) -> str:
    return html.escape(value) if value is not None else "&mdash;"


def _line_chart_svg(
    recent: list[EvalReport],
    value_of,
    y_fmt,
    y_domain: tuple[float, float] | None,
    label: str,
    color: str = "#3f6fd1",
) -> str:
    """Shared SVG geometry for a single-series line chart over `recent` runs.

    `value_of(report) -> float | None` extracts the plotted value (None
    points are skipped). `y_domain` is (lo, hi); pass None to auto-scale to
    the data's own min/max (with a little padding), which latency/token
    counts need since they aren't naturally bounded like accuracy is.
    """
    points = [(r, value_of(r)) for r in recent]
    points = [(r, v) for r, v in points if v is not None]
    if len(points) < 2:
        return f'<p class="muted">Not enough history yet for {label} (need at least 2 runs with data).</p>'

    width, height, pad = 640, 200, 30
    plot_w, plot_h = width - 2 * pad, height - 2 * pad
    n_points = len(points)

    values = [v for _, v in points]
    if y_domain is None:
        lo, hi = min(values), max(values)
        if lo == hi:
            lo, hi = lo - 1, hi + 1
        margin = (hi - lo) * 0.1
        lo, hi = lo - margin, hi + margin
    else:
        lo, hi = y_domain

    def x_at(i: int) -> float:
        return pad + (i / (n_points - 1)) * plot_w

    def y_at(value: float) -> float:
        return pad + (1 - (value - lo) / (hi - lo)) * plot_h

    coords = [(x_at(i), y_at(v)) for i, (_, v) in enumerate(points)]
    path = " ".join(f"{x:.1f},{y:.1f}" for x, y in coords)

    gridline_values = [lo + (hi - lo) * frac for frac in (0.0, 0.25, 0.5, 0.75, 1.0)]
    gridlines = "".join(
        f'<line x1="{pad}" y1="{y_at(v):.1f}" x2="{width - pad}" y2="{y_at(v):.1f}" '
        f'stroke="var(--grid,#ddd)" stroke-width="1" />'
        f'<text x="4" y="{y_at(v) + 4:.1f}" font-size="10" fill="var(--muted,#888)">{y_fmt(v)}</text>'
        for v in gridline_values
    )
    dots = "".join(
        f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3" fill="{color}">'
        f"<title>{_esc(r.prompt_version)} @ {r.ran_at.isoformat()}: {y_fmt(v)}</title></circle>"
        for (x, y), (r, v) in zip(coords, points)
    )

    return f"""
    <svg viewBox="0 0 {width} {height}" role="img" aria-label="{_esc(label)} over the last {n_points} runs">
      {gridlines}
      <polyline points="{path}" fill="none" stroke="{color}" stroke-width="2" />
      {dots}
    </svg>
    """


def _accuracy_chart_svg(history: list[EvalReport], n: int = DEFAULT_TREND_POINTS) -> str:
    recent = sorted(history, key=lambda r: r.ran_at)[-n:]
    return _line_chart_svg(
        recent,
        value_of=lambda r: r.category_accuracy,
        y_fmt=lambda v: f"{v:.0%}",
        y_domain=(0.0, 1.0),
        label="category accuracy",
    )


def _latency_chart_svg(history: list[EvalReport], n: int = DEFAULT_TREND_POINTS) -> str:
    recent = sorted(history, key=lambda r: r.ran_at)[-n:]
    return _line_chart_svg(
        recent,
        value_of=lambda r: r.avg_latency_ms,
        y_fmt=lambda v: f"{v:.0f}ms",
        y_domain=None,
        label="avg latency",
        color="#8a5cf6",
    )


def _tokens_chart_svg(history: list[EvalReport], n: int = DEFAULT_TREND_POINTS) -> str:
    recent = sorted(history, key=lambda r: r.ran_at)[-n:]
    return _line_chart_svg(
        recent,
        value_of=lambda r: r.total_tokens,
        y_fmt=lambda v: f"{v:.0f}",
        y_domain=None,
        label="total tokens",
        color="#e07b39",
    )


def _runs_table_html(rows: list[tuple[EvalReport, str]]) -> str:
    if not rows:
        return '<p class="muted">No runs recorded yet.</p>'

    table_rows = []
    for report, report_path in rows:
        exists = Path(report_path).exists()
        version_cell = (
            f'<a href="{_esc(Path(report_path).as_uri())}">{_esc(report.prompt_version)}</a>'
            if exists
            else _esc(report.prompt_version)
        )
        table_rows.append(
            "<tr>"
            f"<td>{version_cell}</td>"
            f"<td>{_esc(report.model)}</td>"
            f"<td>{report.ran_at.isoformat()}</td>"
            f"<td>{report.category_accuracy:.1%}</td>"
            f'<td>{f"{report.avg_latency_ms:.0f}ms" if report.avg_latency_ms is not None else "&mdash;"}</td>'
            f'<td>{report.total_tokens if report.total_tokens is not None else "&mdash;"}</td>'
            "</tr>"
        )
    return (
        "<table>"
        "<thead><tr><th>Version</th><th>Model</th><th>Ran at</th><th>Accuracy</th>"
        "<th>Avg latency</th><th>Total tokens</th></tr></thead>"
        f"<tbody>{''.join(table_rows)}</tbody></table>"
    )


def render_dashboard_html(
    history: list[EvalReport],
    recent_rows: list[tuple[EvalReport, str]],
    drift: "DriftReport | None" = None,
) -> str:
    """Build the full dashboard HTML as a string. Pure function, no I/O."""
    drift_section = ""
    if drift is not None:
        color = _SEVERITY_COLOR[drift.severity]
        drift_section = f"""
        <section>
          <h2>Drift status</h2>
          <p><span class="badge" style="background:{color}">{drift.severity}</span> {_esc(drift.message)}</p>
          <p class="muted">{drift.window}-run rolling window, {drift.history_length} run(s) of history available.</p>
        </section>
        """

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Eval history dashboard</title>
<style>
  :root {{ color-scheme: light dark; --grid: #ddd; --muted: #777; }}
  body {{ font-family: -apple-system, Segoe UI, Roboto, sans-serif; max-width: 960px; margin: 2rem auto; padding: 0 1rem;
          background: #fff; color: #1a1a1a; }}
  h1 {{ margin-bottom: 0.2rem; }}
  section {{ margin-top: 2rem; }}
  table {{ width: 100%; border-collapse: collapse; margin-top: 0.5rem; }}
  th, td {{ text-align: left; padding: 0.4rem 0.6rem; border-bottom: 1px solid var(--grid); font-size: 0.9rem; }}
  th {{ color: var(--muted); font-weight: 600; }}
  .badge {{ color: #fff; padding: 0.1rem 0.5rem; border-radius: 999px; font-size: 0.75rem; font-weight: 700;
            text-transform: uppercase; }}
  .muted {{ color: var(--muted); }}
  svg {{ width: 100%; height: auto; background: transparent; }}
  @media (prefers-color-scheme: dark) {{
    body {{ background: #111; color: #eee; }}
    :root {{ --grid: #333; --muted: #999; }}
  }}
</style>
</head>
<body>
  <h1>Eval history dashboard</h1>
  <p class="meta muted">{len(history)} run(s) of history.</p>

  {drift_section}

  <section>
    <h2>Category accuracy over time</h2>
    {_accuracy_chart_svg(history)}
  </section>

  <section>
    <h2>Avg latency over time</h2>
    {_latency_chart_svg(history)}
  </section>

  <section>
    <h2>Total tokens over time</h2>
    {_tokens_chart_svg(history)}
  </section>

  <section>
    <h2>Recent runs</h2>
    {_runs_table_html(recent_rows)}
  </section>
</body>
</html>
"""


def write_dashboard_html(
    history: list[EvalReport],
    recent_rows: list[tuple[EvalReport, str]],
    out_path: str | Path,
    drift: "DriftReport | None" = None,
) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(render_dashboard_html(history, recent_rows, drift), encoding="utf-8")
    return out_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Render a standalone HTML trend dashboard from stored eval history.")
    parser.add_argument("--history-db", default=DEFAULT_DB_PATH, help=f"Path to the SQLite history DB (default: {DEFAULT_DB_PATH}).")
    parser.add_argument("--out", default="reports/dashboard.html", help="Path to write the dashboard HTML to.")
    parser.add_argument("--drift-window", type=int, default=None, help="Rolling-average window size in runs (default: drift.DEFAULT_DRIFT_WINDOW).")
    parser.add_argument("--recent", type=int, default=DEFAULT_RECENT_ROWS, help="Number of most-recent runs to show in the table (default: 20).")
    args = parser.parse_args(argv)

    history = load_history(args.history_db)
    recent_rows = load_recent_run_rows(args.recent, args.history_db)

    drift = None
    if history:
        from regression_detector.drift import DEFAULT_DRIFT_WINDOW, detect_drift

        drift = detect_drift(history, window=args.drift_window or DEFAULT_DRIFT_WINDOW)

    out_path = write_dashboard_html(history, recent_rows, args.out, drift)
    print(f"Dashboard written to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
