"""Renders a standalone HTML diff report for one eval run.

The report is a single self-contained .html file (inline CSS, an inline SVG
trend chart, no external assets) so it can be opened locally, attached to a
CI artifact, or hosted anywhere and linked to from a Slack alert.

Contains:
  - run metadata (prompt version, model, timestamp) for baseline + candidate
  - a summary scorecard (candidate vs baseline, every scoring dimension)
  - a side-by-side table of every regressed case (old output vs new output)
  - a trend chart of category accuracy over the last N runs, with the
    rolling average overlaid if a DriftReport is supplied
"""

from __future__ import annotations

import html
from pathlib import Path
from typing import TYPE_CHECKING

from regression_detector.eval_runner import EvalReport

if TYPE_CHECKING:  # avoid a circular import - regression.py imports write_diff_report_html from here
    from regression_detector.drift import DriftReport
    from regression_detector.regression import RegressionReport

_SEVERITY_COLOR = {
    "none": "#2e7d32",
    "warning": "#b26a00",
    "critical": "#c62828",
}


def _esc(value: str | None) -> str:
    return html.escape(value) if value is not None else "&mdash;"


def _status(diff: RegressionReport) -> tuple[str, str]:
    """Overall pass/warn/fail status for the report header."""
    if diff.is_regression and diff.accuracy_severity == "critical":
        return "FAIL", "#c62828"
    if diff.is_regression:
        return "WARN", "#b26a00"
    return "PASS", "#2e7d32"


def _scorecard_row(label: str, baseline, candidate, delta, fmt="{:.1%}", higher_is_better: bool = True) -> str:
    if baseline is None or candidate is None:
        return f"<tr><td>{label}</td><td>&mdash;</td><td>&mdash;</td><td>&mdash;</td></tr>"
    good = (delta >= 0) if higher_is_better else (delta <= 0)
    delta_color = "#2e7d32" if good or delta == 0 else "#c62828"
    sign = "+" if delta > 0 else ""
    return (
        f"<tr><td>{label}</td>"
        f"<td>{fmt.format(baseline)}</td>"
        f"<td>{fmt.format(candidate)}</td>"
        f'<td style="color:{delta_color};font-weight:600">{sign}{fmt.format(delta)}</td></tr>'
    )


def _regressed_rows(diff: RegressionReport) -> str:
    regressed = [d for d in diff.example_diffs if d.category_regressed]
    if not regressed:
        return '<p class="muted">No regressed cases in this run.</p>'

    rows = []
    for d in regressed:
        rows.append(
            "<tr>"
            f"<td>{_esc(d.id)}</td>"
            f"<td>{_esc(d.email)}</td>"
            f"<td>{_esc(d.expected_category)}</td>"
            f'<td class="old">{_esc(d.baseline_category)}<br><span class="summary">{_esc(d.baseline_summary)}</span></td>'
            f'<td class="new">{_esc(d.candidate_category)}<br><span class="summary">{_esc(d.candidate_summary)}</span></td>'
            "</tr>"
        )
    return (
        "<table class='regressions'>"
        "<thead><tr><th>ID</th><th>Email</th><th>Expected</th>"
        "<th>Baseline (old)</th><th>Candidate (new)</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def _category_delta_rows(diff: RegressionReport) -> str:
    rows = []
    for cd in diff.category_deltas:
        color = _SEVERITY_COLOR[cd.severity]
        sign = "+" if cd.delta > 0 else ""
        rows.append(
            "<tr>"
            f"<td>{_esc(cd.category)}</td>"
            f"<td>{cd.baseline_accuracy:.1%}</td>"
            f"<td>{cd.candidate_accuracy:.1%}</td>"
            f'<td style="color:{color};font-weight:600">{sign}{cd.delta:.1%}</td>'
            f'<td><span class="badge" style="background:{color}">{cd.severity}</span></td>'
            "</tr>"
        )
    return "".join(rows)


def _trend_chart_svg(history: list[EvalReport], drift: DriftReport | None, n: int = 20) -> str:
    recent = sorted(history, key=lambda r: r.ran_at)[-n:]
    if len(recent) < 2:
        return '<p class="muted">Not enough history yet for a trend chart (need at least 2 runs).</p>'

    width, height, pad = 640, 200, 30
    plot_w, plot_h = width - 2 * pad, height - 2 * pad
    n_points = len(recent)

    def x_at(i: int) -> float:
        return pad + (i / (n_points - 1)) * plot_w

    def y_at(value: float) -> float:
        # Accuracy is 0..1; give a little headroom above/below for legibility.
        lo, hi = 0.0, 1.0
        return pad + (1 - (value - lo) / (hi - lo)) * plot_h

    accuracy_points = [(x_at(i), y_at(r.category_accuracy)) for i, r in enumerate(recent)]
    accuracy_path = " ".join(f"{x:.1f},{y:.1f}" for x, y in accuracy_points)

    rolling_path = ""
    if drift is not None:
        drift_by_time = {p.ran_at: p.rolling_avg for p in drift.points}
        rolling_points = [
            (x_at(i), y_at(drift_by_time[r.ran_at]))
            for i, r in enumerate(recent)
            if r.ran_at in drift_by_time and drift_by_time[r.ran_at] is not None
        ]
        if len(rolling_points) >= 2:
            rolling_path = " ".join(f"{x:.1f},{y:.1f}" for x, y in rolling_points)

    gridlines = "".join(
        f'<line x1="{pad}" y1="{y_at(v):.1f}" x2="{width - pad}" y2="{y_at(v):.1f}" '
        f'stroke="var(--grid,#ddd)" stroke-width="1" />'
        f'<text x="4" y="{y_at(v) + 4:.1f}" font-size="10" fill="var(--muted,#888)">{v:.0%}</text>'
        for v in (0.0, 0.25, 0.5, 0.75, 1.0)
    )
    dots = "".join(
        f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3" fill="#3f6fd1">'
        f"<title>{_esc(r.prompt_version)} @ {r.ran_at.isoformat()}: {r.category_accuracy:.1%}</title></circle>"
        for (x, y), r in zip(accuracy_points, recent)
    )
    rolling_svg = (
        f'<polyline points="{rolling_path}" fill="none" stroke="#c62828" stroke-width="2" '
        f'stroke-dasharray="5,4" />'
        if rolling_path
        else ""
    )

    return f"""
    <svg viewBox="0 0 {width} {height}" role="img" aria-label="Category accuracy over the last {n_points} runs">
      {gridlines}
      <polyline points="{accuracy_path}" fill="none" stroke="#3f6fd1" stroke-width="2" />
      {rolling_svg}
      {dots}
    </svg>
    <p class="legend">
      <span class="swatch" style="background:#3f6fd1"></span> category accuracy per run
      {'<span class="swatch dashed" style="border-color:#c62828"></span> rolling average' if rolling_svg else ''}
    </p>
    """


def render_diff_report_html(
    baseline: EvalReport,
    candidate: EvalReport,
    diff: RegressionReport,
    history: list[EvalReport] | None = None,
    drift: DriftReport | None = None,
) -> str:
    """Build the full HTML report as a string."""
    history = history or [baseline, candidate]
    status_label, status_color = _status(diff)

    scorecard = "".join(
        [
            _scorecard_row("Category accuracy", baseline.category_accuracy, candidate.category_accuracy, diff.accuracy_delta),
            _scorecard_row(
                "Summary score (heuristic)",
                baseline.avg_summary_score,
                candidate.avg_summary_score,
                diff.avg_summary_score_delta,
            ),
            _scorecard_row(
                "Summary score (LLM judge, 1-5)",
                baseline.avg_summary_judge_score,
                candidate.avg_summary_judge_score,
                (
                    round(candidate.avg_summary_judge_score - baseline.avg_summary_judge_score, 3)
                    if baseline.avg_summary_judge_score is not None and candidate.avg_summary_judge_score is not None
                    else None
                ),
                fmt="{:.2f}",
            ),
            _scorecard_row(
                "Avg latency (ms)",
                baseline.avg_latency_ms,
                candidate.avg_latency_ms,
                (
                    round(candidate.avg_latency_ms - baseline.avg_latency_ms, 1)
                    if baseline.avg_latency_ms is not None and candidate.avg_latency_ms is not None
                    else None
                ),
                fmt="{:.1f}",
                higher_is_better=False,
            ),
            _scorecard_row(
                "Total tokens",
                baseline.total_tokens,
                candidate.total_tokens,
                (
                    candidate.total_tokens - baseline.total_tokens
                    if baseline.total_tokens is not None and candidate.total_tokens is not None
                    else None
                ),
                fmt="{:d}",
                higher_is_better=False,
            ),
        ]
    )

    drift_section = ""
    if drift is not None:
        drift_color = _SEVERITY_COLOR[drift.severity]
        drift_section = f"""
        <section>
          <h2>Drift detection</h2>
          <p>
            <span class="badge" style="background:{drift_color}">{drift.severity}</span>
            {_esc(drift.message)}
          </p>
          <p class="muted">{drift.window}-run rolling window, {drift.history_length} run(s) of history available.</p>
        </section>
        """

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Eval diff: {_esc(baseline.prompt_version)} vs {_esc(candidate.prompt_version)}</title>
<style>
  :root {{ color-scheme: light dark; --grid: #ddd; --muted: #777; }}
  body {{ font-family: -apple-system, Segoe UI, Roboto, sans-serif; max-width: 960px; margin: 2rem auto; padding: 0 1rem;
          background: #fff; color: #1a1a1a; }}
  h1 {{ margin-bottom: 0.2rem; }}
  .status {{ display: inline-block; padding: 0.25rem 0.75rem; border-radius: 6px; color: #fff; font-weight: 700;
             letter-spacing: 0.04em; }}
  .meta {{ color: var(--muted); margin-top: 0.25rem; }}
  section {{ margin-top: 2rem; }}
  table {{ width: 100%; border-collapse: collapse; margin-top: 0.5rem; }}
  th, td {{ text-align: left; padding: 0.4rem 0.6rem; border-bottom: 1px solid var(--grid); font-size: 0.9rem; }}
  th {{ color: var(--muted); font-weight: 600; }}
  .regressions .old {{ background: rgba(198,40,40,0.08); }}
  .regressions .new {{ background: rgba(46,125,50,0.08); }}
  .regressions .summary {{ color: var(--muted); font-size: 0.85rem; }}
  .badge {{ color: #fff; padding: 0.1rem 0.5rem; border-radius: 999px; font-size: 0.75rem; font-weight: 700;
            text-transform: uppercase; }}
  .muted {{ color: var(--muted); }}
  .legend {{ font-size: 0.85rem; color: var(--muted); }}
  .swatch {{ display: inline-block; width: 12px; height: 12px; background: #3f6fd1; margin-right: 4px; vertical-align: middle; }}
  .swatch.dashed {{ background: none; border: 2px dashed; }}
  svg {{ width: 100%; height: auto; background: transparent; }}
  @media (prefers-color-scheme: dark) {{
    body {{ background: #111; color: #eee; }}
    :root {{ --grid: #333; --muted: #999; }}
  }}
</style>
</head>
<body>
  <h1>Eval diff report <span class="status" style="background:{status_color}">{status_label}</span></h1>
  <p class="meta">
    Baseline: <strong>{_esc(baseline.prompt_version)}</strong> ({_esc(baseline.model)}) @ {baseline.ran_at.isoformat()}<br>
    Candidate: <strong>{_esc(candidate.prompt_version)}</strong> ({_esc(candidate.model)}) @ {candidate.ran_at.isoformat()}<br>
    Compared at {diff.compared_at.isoformat()}
  </p>

  <section>
    <h2>Scorecard</h2>
    <table>
      <thead><tr><th></th><th>Baseline</th><th>Candidate</th><th>Delta</th></tr></thead>
      <tbody>{scorecard}</tbody>
    </table>
  </section>

  <section>
    <h2>Per-category accuracy</h2>
    <table>
      <thead><tr><th>Category</th><th>Baseline</th><th>Candidate</th><th>Delta</th><th>Severity</th></tr></thead>
      <tbody>{_category_delta_rows(diff)}</tbody>
    </table>
  </section>

  <section>
    <h2>Regressed cases ({len(diff.regressed_examples)})</h2>
    {_regressed_rows(diff)}
  </section>

  <section>
    <h2>Trend (last runs)</h2>
    {_trend_chart_svg(history, drift)}
  </section>

  {drift_section}
</body>
</html>
"""


def write_diff_report_html(
    baseline: EvalReport,
    candidate: EvalReport,
    diff: RegressionReport,
    out_path: str | Path,
    history: list[EvalReport] | None = None,
    drift: DriftReport | None = None,
) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(render_diff_report_html(baseline, candidate, diff, history, drift), encoding="utf-8")
    return out_path
