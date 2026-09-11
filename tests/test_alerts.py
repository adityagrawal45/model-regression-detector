from datetime import datetime, timezone

import pytest

from regression_detector.alerts import SlackAlertError, build_slack_message, send_slack_alert
from regression_detector.drift import DriftReport
from regression_detector.regression import CategoryDelta, RegressionReport


def _diff(accuracy_delta=-0.10, is_regression=True, accuracy_severity="critical", regressed=None, category_deltas=None):
    return RegressionReport(
        baseline_version="v1",
        candidate_version="v2",
        baseline_model="m",
        candidate_model="m",
        compared_at=datetime.now(timezone.utc),
        baseline_accuracy=0.94,
        candidate_accuracy=0.94 + accuracy_delta,
        accuracy_delta=accuracy_delta,
        accuracy_severity=accuracy_severity,
        avg_summary_score_delta=0.0,
        category_deltas=category_deltas or [],
        warning_delta_threshold=0.03,
        critical_delta_threshold=0.08,
        regressed_examples=regressed or [],
        improved_examples=[],
        new_errors=[],
        is_regression=is_regression,
        reasons=["something regressed"] if is_regression else [],
        example_diffs=[],
    )


def test_build_slack_message_fail_status_and_headline_numbers():
    diff = _diff(accuracy_delta=-0.05, regressed=["a", "b", "c"])

    payload = build_slack_message(diff, report_url="https://example.com/report.html")

    assert ":x:" in payload["text"] or "FAIL" in payload["text"]
    header_text = payload["blocks"][0]["text"]["text"]
    assert "FAIL" in header_text
    body = " ".join(str(b) for b in payload["blocks"])
    assert "3 regression" in body.replace("Regressions:", "3 regression") or "Regressions" in body
    assert "94%" in body
    assert "89%" in body
    assert "https://example.com/report.html" in body


def test_build_slack_message_pass_status_when_no_regression():
    diff = _diff(accuracy_delta=0.0, is_regression=False, accuracy_severity="none")

    payload = build_slack_message(diff)

    header_text = payload["blocks"][0]["text"]["text"]
    assert "PASS" in header_text


def test_build_slack_message_warn_status_for_non_critical_regression():
    diff = _diff(accuracy_delta=-0.04, is_regression=True, accuracy_severity="warning")

    payload = build_slack_message(diff)

    header_text = payload["blocks"][0]["text"]["text"]
    assert "WARN" in header_text


def test_build_slack_message_includes_category_delta_lines():
    category_deltas = [
        CategoryDelta(category="technical", baseline_accuracy=0.9, candidate_accuracy=0.6, delta=-0.3, severity="critical"),
        CategoryDelta(category="billing", baseline_accuracy=0.9, candidate_accuracy=0.9, delta=0.0, severity="none"),
    ]
    diff = _diff(category_deltas=category_deltas)

    payload = build_slack_message(diff)

    body = " ".join(str(b) for b in payload["blocks"])
    assert "technical" in body
    assert "billing" not in body  # only non-none severities get a line


def test_build_slack_message_includes_drift_line_when_drifting():
    diff = _diff(accuracy_delta=0.0, is_regression=False, accuracy_severity="none")
    drift = DriftReport(
        window=7,
        history_length=14,
        current_rolling_avg=0.80,
        reference_rolling_avg=0.93,
        drift_delta=-0.13,
        severity="critical",
        is_drift=True,
        message="[CRITICAL] Slow drift detected: 7-run rolling average moved -13.0pp.",
    )

    payload = build_slack_message(diff, drift=drift)

    body = " ".join(str(b) for b in payload["blocks"])
    assert "Drift" in body
    assert "Slow drift detected" in body


def test_send_slack_alert_uses_injected_sender_without_network():
    diff = _diff()
    captured = {}

    def fake_sender(url, payload):
        captured["url"] = url
        captured["payload"] = payload

    result = send_slack_alert(diff, webhook_url="https://hooks.slack.test/abc", sender=fake_sender)

    assert captured["url"] == "https://hooks.slack.test/abc"
    assert captured["payload"] == result


def test_send_slack_alert_raises_without_webhook_url(monkeypatch):
    monkeypatch.delenv("SLACK_WEBHOOK_URL", raising=False)
    diff = _diff()

    with pytest.raises(SlackAlertError):
        send_slack_alert(diff, sender=lambda url, payload: None)


def test_send_slack_alert_falls_back_to_env_var(monkeypatch):
    monkeypatch.setenv("SLACK_WEBHOOK_URL", "https://hooks.slack.test/from-env")
    diff = _diff()
    captured = {}

    send_slack_alert(diff, sender=lambda url, payload: captured.setdefault("url", url))

    assert captured["url"] == "https://hooks.slack.test/from-env"
