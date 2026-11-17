"""Descriptive analysis that keeps unresolved trials in the denominator."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from math import inf, isfinite, sqrt
from typing import Literal

CensoringRule = Literal["failure_as_infinity", "right_censored"]
PopulationConclusion = Literal[
    "finite_supported",
    "infinite_supported",
    "not_identifiable_supported",
    "indeterminate",
]


@dataclass(frozen=True)
class TrialSummary:
    total_trials: int
    resolved_trials: int
    resolution_rate: float
    unresolved_rate: float
    unresolved_rate_ci_low: float
    unresolved_rate_ci_high: float
    conditional_p90_ms: float | None
    unconditional_p90_ms: float | None
    unconditional_p90_status: Literal["finite", "infinite", "not_identifiable"]
    population_p90_conclusion: PopulationConclusion


def unresolved_rate(success_probability: float, attempts: int = 2) -> float:
    """Return the unresolved share under independent, equal-probability attempts."""
    if not 0 <= success_probability <= 1:
        raise ValueError("success_probability must be between 0 and 1")
    if attempts < 1:
        raise ValueError("attempts must be at least 1")
    return (1 - success_probability) ** attempts


def percentile(values: Iterable[float], quantile: float) -> float | None:
    """Return a linearly interpolated sample quantile."""
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return None
    if not 0 <= quantile <= 1:
        raise ValueError("quantile must be between 0 and 1")
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    if fraction == 0:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Return a two-sided Wilson score interval for a binomial proportion."""
    if total < 1:
        raise ValueError("total must be positive")
    if not 0 <= successes <= total:
        raise ValueError("successes must be between zero and total")
    proportion = successes / total
    z_squared = z * z
    denominator = 1 + z_squared / total
    center = (proportion + z_squared / (2 * total)) / denominator
    spread = (
        z
        * sqrt(
            proportion * (1 - proportion) / total
            + z_squared / (4 * total * total)
        )
        / denominator
    )
    return max(0.0, center - spread), min(1.0, center + spread)


def summarize_trials(
    trials: Iterable[Mapping[str, object]],
    *,
    censoring: CensoringRule,
) -> TrialSummary:
    """Summarize resolution and latency without dropping failed tasks."""
    rows = list(trials)
    if not rows:
        raise ValueError("at least one trial is required")
    if censoring not in ("failure_as_infinity", "right_censored"):
        raise ValueError(f"unsupported censoring rule: {censoring}")

    durations: list[float] = []
    for row in rows:
        if not bool(row.get("resolved")):
            continue
        duration = row.get("duration_ms")
        if (
            isinstance(duration, bool)
            or not isinstance(duration, (int, float))
            or not isfinite(float(duration))
        ):
            raise ValueError("resolved trials require a finite duration_ms")
        durations.append(float(duration))

    resolved = len(durations)
    total = len(rows)
    resolution = resolved / total
    unresolved_count = total - resolved
    unresolved = unresolved_count / total
    ci_low, ci_high = wilson_interval(unresolved_count, total)

    if unresolved >= 0.10:
        status: Literal["finite", "infinite", "not_identifiable"]
        status = "infinite" if censoring == "failure_as_infinity" else "not_identifiable"
        unconditional_p90 = None
    else:
        status = "finite"
        unconditional_p90 = percentile(
            [*durations, *([inf] * unresolved_count)],
            0.90,
        )

    if ci_high < 0.10:
        population_conclusion: PopulationConclusion = "finite_supported"
    elif ci_low >= 0.10:
        population_conclusion = (
            "infinite_supported"
            if censoring == "failure_as_infinity"
            else "not_identifiable_supported"
        )
    else:
        population_conclusion = "indeterminate"

    return TrialSummary(
        total_trials=total,
        resolved_trials=resolved,
        resolution_rate=resolution,
        unresolved_rate=unresolved,
        unresolved_rate_ci_low=ci_low,
        unresolved_rate_ci_high=ci_high,
        conditional_p90_ms=percentile(durations, 0.90),
        unconditional_p90_ms=unconditional_p90,
        unconditional_p90_status=status,
        population_p90_conclusion=population_conclusion,
    )
