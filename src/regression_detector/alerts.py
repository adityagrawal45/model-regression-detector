"""Slack alerting via an incoming webhook.

Sends a structured message (Block Kit) with a pass/warn/fail status line, the
headline numbers, and a link to the full HTML diff report. The HTTP POST
itself is a single small function (`_post_to_slack`) injected as a `sender`
callable everywhere else, so tests can verify the built payload without ever
making a network call.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import TYPE_CHECKING, Callable, Literal

if TYPE_CHECKING:  # avoid a circular import - regression.py imports send_slack_alert from here
    from regression_detector.drift import DriftReport
    from regression_detector.regression import RegressionReport

Status = Literal["PASS", "WARN", "FAIL"]

_STATUS_EMOJI = {"PASS": ":white_check_mark:", "WARN": ":warning:", "FAIL": ":x:"}

Sender = Callable[[str, dict], None]


class SlackAlertError(Exception):
    """Raised when posting to the Slack webhook fails."""


def _status_for(diff: RegressionReport) -> Status:
    if diff.is_regression and diff.accuracy_severity == "critical":
        return "FAIL"
    if diff.is_regression:
        return "WARN"
    return "PASS"


def build_slack_message(
    diff: RegressionReport,
    report_url: str | None = None,
    drift: DriftReport | None = None,
) -> dict:
    """Build the Slack Block Kit payload for a regression diff (and optional drift check)."""
    status = _status_for(diff)
    emoji = _STATUS_EMOJI[status]

    headline = (
        f"{emoji} *{status}* — {len(diff.regressed_examples)} regression(s) detected, accuracy "
        f"{'dropped' if diff.accuracy_delta < 0 else 'moved'} from {diff.baseline_accuracy:.0%} "
        f"to {diff.candidate_accuracy:.0%} ({diff.candidate_version} vs {diff.baseline_version})."
    )

    lines = [
        f"*Regressions:* {len(diff.regressed_examples)}   *Improvements:* {len(diff.improved_examples)}"
        f"   *New errors:* {len(diff.new_errors)}",
        f"*Overall pass rate delta:* {diff.accuracy_delta:+.1%}  ({diff.accuracy_severity})",
    ]
    for cd in diff.category_deltas:
        if cd.severity != "none":
            lines.append(f"  • `{cd.category}` {cd.baseline_accuracy:.0%} → {cd.candidate_accuracy:.0%} ({cd.severity})")

    if drift is not None and drift.is_drift:
        lines.append(f":chart_with_downwards_trend: *Drift:* {drift.message}")

    blocks: list[dict] = [
        {"type": "header", "text": {"type": "plain_text", "text": f"{emoji} Eval run: {status}"}},
        {"type": "section", "text": {"type": "mrkdwn", "text": headline}},
        {"type": "section", "text": {"type": "mrkdwn", "text": "\n".join(lines)}},
    ]
    if report_url:
        blocks.append(
            {
                "type": "section",
                "text": {"type": "mrkdwn", "text": f"<{report_url}|View full HTML diff report>"},
            }
        )

    # `text` is the fallback for notifications/screen readers when blocks
    # aren't rendered (e.g. some third-party Slack-compatible endpoints).
    fallback_text = f"{emoji} {status}: {diff.candidate_version} vs {diff.baseline_version} - {headline}"
    return {"text": fallback_text, "blocks": blocks}


def _post_to_slack(webhook_url: str, payload: dict) -> None:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        webhook_url, data=body, headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            if response.status >= 300:
                raise SlackAlertError(f"Slack webhook returned status {response.status}")
    except urllib.error.URLError as exc:
        raise SlackAlertError(f"Failed to reach Slack webhook: {exc}") from exc


def send_slack_alert(
    diff: RegressionReport,
    webhook_url: str | None = None,
    report_url: str | None = None,
    drift: DriftReport | None = None,
    sender: Sender = _post_to_slack,
) -> dict:
    """Build and send the Slack alert for a regression diff. Returns the payload sent.

    `webhook_url` falls back to the SLACK_WEBHOOK_URL env var. `sender` is
    injectable (defaults to the real HTTP POST) so callers/tests can capture
    the payload without hitting the network.
    """
    webhook_url = webhook_url or os.environ.get("SLACK_WEBHOOK_URL")
    if not webhook_url:
        raise SlackAlertError(
            "No Slack webhook URL provided. Pass --slack-webhook-url or set SLACK_WEBHOOK_URL."
        )
    payload = build_slack_message(diff, report_url=report_url, drift=drift)
    sender(webhook_url, payload)
    return payload


__all__ = [
    "SlackAlertError",
    "Status",
    "build_slack_message",
    "send_slack_alert",
]
