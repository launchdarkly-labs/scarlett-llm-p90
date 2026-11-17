"""Keep the published result summary tied to its checked-in source rows."""

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, median

import pytest

from llm_p90.analysis import percentile, wilson_interval

ROOT = Path(__file__).resolve().parents[1]


def test_published_results_recompute_from_the_checked_in_log():
    log_path = ROOT / "logs" / "confirmatory.jsonl"
    published = json.loads((ROOT / "data" / "confirmatory_results.json").read_text())
    rows = [
        json.loads(line)
        for line in log_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert hashlib.sha256(log_path.read_bytes()).hexdigest() == (
        published["provenance"]["confirmatory_log_sha256"]
    )
    assert len(rows) == published["experiment"]["assigned_total"] == 170
    assert len({row["request_id"] for row in rows}) == len(rows)

    by_variation = defaultdict(list)
    for row in rows:
        by_variation[row["variation"]].append(row)

    for variation, expected in published["variations"].items():
        arm = by_variation[variation]
        resolved = [row for row in arm if row["resolved"]]
        durations = [row["duration_ms"] for row in resolved]
        unresolved_count = len(arm) - len(resolved)

        assert len(arm) == expected["assigned"]
        assert len(resolved) == expected["resolved"]
        assert sum(row["resolved"] and row["attempts"] == 1 for row in arm) == (
            expected["resolved_first_attempt"]
        )
        assert sum(row["resolved"] and row["attempts"] == 2 for row in arm) == (
            expected["resolved_second_attempt"]
        )
        assert sum(row["attempts"] == 2 for row in arm) == expected["second_attempts"]
        assert len(resolved) / len(arm) == pytest.approx(expected["resolution_rate"])
        assert wilson_interval(unresolved_count, len(arm)) == pytest.approx(
            expected["unresolved_rate_wilson_95"]
        )
        assert mean(durations) == pytest.approx(expected["mean_duration_ms_among_resolved"])
        assert median(durations) == pytest.approx(
            expected["median_duration_ms_among_resolved"]
        )
        assert percentile(durations, 0.90) == pytest.approx(
            expected["type7_p90_duration_ms_among_resolved"]
        )
        for seconds, count in expected["resolved_within_seconds"].items():
            assert sum(duration <= int(seconds) * 1000 for duration in durations) == count


def test_launchdarkly_snapshot_pins_the_remote_inputs():
    snapshot = json.loads((ROOT / "data" / "launchdarkly_snapshot.json").read_text())
    published = json.loads((ROOT / "data" / "confirmatory_results.json").read_text())
    tools_path = ROOT / "tools" / "bfcl_tools.json"

    assert snapshot["experiment"]["iteration_id"] == published["experiment"]["iteration_id"]
    assert snapshot["config"]["tool_definitions_sha256"] == hashlib.sha256(
        tools_path.read_bytes()
    ).hexdigest()
    assert len(json.loads(tools_path.read_text())["tools"]) == 128
    assert snapshot["config"]["targeting"]["allocation"] == {
        "claude-sonnet-5-5": 50,
        "claude-haiku-5-5": 50,
    }
    assert snapshot["metrics"]["task-resolution-duration-p90"] == {
        "version_id": "f70e55cc-21ca-4c76-9ce2-0435d0a9110f",
        "event_key": "task-resolution-duration-ms",
        "measure_type": "value",
        "analysis_type": "percentile",
        "percentile_value": 90,
        "analysis_unit": "request",
        "unit_aggregation": "average",
        "unit": "ms",
        "units_without_events": "excluded",
        "success_criteria": "LowerThanBaseline",
    }
