import json
from hashlib import sha256
from pathlib import Path

import pytest

from llm_p90.analysis import percentile, summarize_trials, unresolved_rate
from llm_p90.protocol import ExperimentProtocol, TrialState
from llm_p90.scoring import evaluate_with_bfcl, score_with_bfcl


def test_unresolved_rate_uses_the_full_attempt_budget():
    assert unresolved_rate(0.5624, attempts=2) == pytest.approx(0.19149376)
    assert unresolved_rate(0.7324, attempts=2) == pytest.approx(0.07160976)


def test_summary_never_drops_unresolved_trials():
    summary = summarize_trials(
        [
            {"resolved": True, "duration_ms": 1000},
            {"resolved": True, "duration_ms": 2000},
            {"resolved": False, "duration_ms": None},
            {"resolved": False, "duration_ms": None},
        ],
        censoring="failure_as_infinity",
    )

    assert summary.total_trials == 4
    assert summary.resolved_trials == 2
    assert summary.resolution_rate == 0.5
    assert summary.conditional_p90_ms == pytest.approx(1900)
    assert summary.unconditional_p90_ms is None
    assert summary.unconditional_p90_status == "infinite"


def test_right_censoring_makes_unconditional_p90_not_identifiable():
    trials = [
        *({"resolved": True, "duration_ms": index + 1} for index in range(80)),
        *({"resolved": False, "duration_ms": None} for _ in range(20)),
    ]

    summary = summarize_trials(trials, censoring="right_censored")

    assert summary.unresolved_rate == 0.2
    assert summary.unconditional_p90_status == "not_identifiable"


def test_finite_unconditional_p90_is_returned_when_more_than_90_percent_resolve():
    trials = [
        *({"resolved": True, "duration_ms": index + 1} for index in range(91)),
        *({"resolved": False, "duration_ms": None} for _ in range(9)),
    ]

    summary = summarize_trials(trials, censoring="failure_as_infinity")

    assert summary.unconditional_p90_status == "finite"
    assert summary.conditional_p90_ms == pytest.approx(82.0)
    assert summary.unconditional_p90_ms == pytest.approx(90.1)


def test_exactly_10_percent_unresolved_is_infinite_under_type_7_interpolation():
    trials = [
        *({"resolved": True, "duration_ms": index + 1} for index in range(90)),
        *({"resolved": False, "duration_ms": None} for _ in range(10)),
    ]

    summary = summarize_trials(trials, censoring="failure_as_infinity")

    assert summary.unconditional_p90_status == "infinite"
    assert summary.unconditional_p90_ms is None


def test_sample_uncertainty_is_reported_around_the_10_percent_boundary():
    trials = [
        *({"resolved": True, "duration_ms": index + 1} for index in range(139)),
        *({"resolved": False, "duration_ms": None} for _ in range(11)),
    ]

    summary = summarize_trials(trials, censoring="failure_as_infinity")

    assert summary.unresolved_rate == pytest.approx(11 / 150)
    assert summary.unresolved_rate_ci_low < 0.10 < summary.unresolved_rate_ci_high
    assert summary.population_p90_conclusion == "indeterminate"


def test_boolean_duration_is_rejected():
    with pytest.raises(ValueError, match="finite duration_ms"):
        summarize_trials(
            [{"resolved": True, "duration_ms": True}],
            censoring="failure_as_infinity",
        )


def test_trial_state_allows_only_one_repair_attempt():
    state = TrialState(max_attempts=2)

    assert state.begin_attempt() == 1
    assert state.begin_attempt() == 2
    with pytest.raises(RuntimeError, match="attempt budget exhausted"):
        state.begin_attempt()


def test_protocol_rejects_unregistered_timing_choices():
    with pytest.raises(ValueError, match="streaming"):
        ExperimentProtocol(
            runner_region="us-east-1",
            concurrency=1,
            streaming=None,
            timeout_seconds=60,
            temperature=0,
            max_tokens=512,
            prompt_caching=False,
            warmup_rows=2,
            retry_429=False,
        )


def test_deterministic_scorer_delegates_to_the_official_bfcl_evaluator():
    calls = []

    def official_evaluator(expected, generated):
        calls.append((expected, generated))
        return {"valid": True}

    row = {"expected_output": json.dumps(["lookup_order(order_id='A-17')"])}
    output = json.dumps({"calls": [{"name": "lookup_order", "arguments": {"order_id": "A-17"}}]})

    assert score_with_bfcl(row, output, evaluator=official_evaluator) == 1.0
    assert len(calls) == 1


def test_official_bfcl_multi_turn_evaluator_accepts_ground_truth_trajectory():
    root = Path(__file__).parents[1]
    row = json.loads(
        (root / "data" / "bfcl_pilot.jsonl").read_text().splitlines()[0]
    )
    expected = json.loads(row["expected_output"])
    output = json.dumps({"turns": [[turn] for turn in expected]})

    result = evaluate_with_bfcl(
        row,
        output,
        model_name="claude-sonnet-4-5-20250929",
    )

    assert result["valid"] is True


def test_source_snapshot_has_pinned_provenance():
    root = Path(__file__).parents[1]
    metadata = json.loads((root / "data" / "provenance.json").read_text())
    source = root / "data" / metadata["leaderboard"]["file"]

    assert metadata["leaderboard"]["commit"] == "ea03bc8dd716546f4d4c6dd462368a9ab18cc2e0"
    assert metadata["leaderboard"]["evaluation_checkpoint"] == "f7cf735"
    assert len(metadata["tasks"]["commit"]) == 40
    assert metadata["retrieved_at"]
    assert sha256(source.read_bytes()).hexdigest() == metadata["leaderboard"]["sha256"]

    pilot = root / "data" / metadata["tasks"]["pilot"]["file"]
    confirmatory = root / "data" / metadata["tasks"]["confirmatory"]["file"]
    assert sha256(pilot.read_bytes()).hexdigest() == metadata["tasks"]["pilot"]["sha256"]
    assert (
        sha256(confirmatory.read_bytes()).hexdigest()
        == metadata["tasks"]["confirmatory"]["sha256"]
    )


def test_pilot_and_confirmatory_tasks_are_disjoint():
    root = Path(__file__).parents[1]

    def ids(path):
        return {
            json.loads(line)["metadata"]["source_id"]
            for line in path.read_text().splitlines()
            if line
        }

    pilot = ids(root / "data" / "bfcl_pilot.jsonl")
    confirmatory = ids(root / "data" / "bfcl_confirmatory.jsonl")

    assert len(pilot) == 30
    assert len(confirmatory) == 170
    assert pilot.isdisjoint(confirmatory)


def test_percentile_at_an_exact_rank_ignores_the_next_value():
    assert percentile([1.0, 2.0, float("inf")], 0.5) == 2.0


def test_evaluate_with_bfcl_reports_bad_metadata_as_invalid():
    row = {"expected_output": "[]", "metadata": "not a mapping"}

    result = evaluate_with_bfcl(row, json.dumps({"turns": []}))

    assert result["valid"] is False


def test_protocol_file_passes_protocol_validation():
    root = Path(__file__).parents[1]
    execution = json.loads((root / "protocol.json").read_text())["execution"]

    protocol = ExperimentProtocol(
        runner_region=execution["runner_region"],
        concurrency=execution["concurrency"],
        streaming=execution["streaming"],
        timeout_seconds=execution["timeout_seconds"],
        temperature=execution["temperature"],
        max_tokens=execution["max_tokens"],
        prompt_caching=execution["prompt_caching"],
        warmup_rows=execution["warmup_rows_per_model"],
        retry_429=execution["retry_429"],
    )

    assert protocol.max_tokens == execution["max_tokens"]
