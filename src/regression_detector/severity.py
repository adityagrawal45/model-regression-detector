"""Shared statistical-significance severity scale.

Used by both `regression.py` (per-run, per-category accuracy deltas) and
`drift.py` (rolling-average trend deltas) so "is this delta noise, a
warning, or critical" means the same thing - and is tuned the same way -
everywhere in the pipeline.
"""

from __future__ import annotations

from typing import Literal

# Default statistical-significance thresholds. A delta whose absolute value
# exceeds `critical` is "critical"; between `warning` and `critical` is
# "warning"; below `warning` is "none" (noise). Tune these to your dataset
# size - a couple of flipped cases on a 20-case dataset is a bigger swing
# than the same count on a 500-case one.
DEFAULT_WARNING_DELTA_THRESHOLD = 0.03
DEFAULT_CRITICAL_DELTA_THRESHOLD = 0.08

Severity = Literal["none", "warning", "critical"]


def classify_severity(
    delta: float,
    warning_threshold: float = DEFAULT_WARNING_DELTA_THRESHOLD,
    critical_threshold: float = DEFAULT_CRITICAL_DELTA_THRESHOLD,
) -> Severity:
    """Classify a drop's magnitude as noise, a warning, or critical.

    Only drops (negative deltas) are ever escalated - an improvement is never
    "critical". Magnitude is compared against the configurable thresholds so
    callers can tune signal-vs-noise to their dataset size.
    """
    drop = -delta
    if drop >= critical_threshold:
        return "critical"
    if drop >= warning_threshold:
        return "warning"
    return "none"


__all__ = [
    "DEFAULT_WARNING_DELTA_THRESHOLD",
    "DEFAULT_CRITICAL_DELTA_THRESHOLD",
    "Severity",
    "classify_severity",
]
