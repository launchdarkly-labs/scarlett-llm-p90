"""Drive both runners end to end with a scripted provider. No network, no spend."""

import asyncio
import io
import json
from types import SimpleNamespace

import pytest
from bfcl_policies import ROOT, TOOL_KEYS, ground_truth_turns, rows
from launchdarkly_ai_server import create_handler
from ldclient import Context

import run_evaluation
import run_experiment
from llm_p90.bfcl_loop import tool_implementations
from llm_p90.scoring import score_with_bfcl


def test_pilot_handler_runs_a_whole_episode_and_the_official_scorer_passes_it(monkeypatch):
    row = rows("bfcl_pilot.jsonl")[0]
    inline = run_evaluation.load_inline_dataset(ROOT / "data" / "bfcl_pilot.jsonl")[0]
    tools = tool_implementations(TOOL_KEYS)
    policy = ground_truth_turns(row, tools)
    calls = []

    async def scripted(config, user_input, tool_handlers, variables, history=None):
        calls.append(user_input)
        # Stands in for the Claude handler's native tool loop.
        output = await policy(user_input, history)
        return {"output": output, "usage": {"total": 10}}

    fake = create_handler(("anthropic", "messages"), scripted)
    monkeypatch.setattr(run_evaluation, "provider_handler", lambda *_: fake)

    handler = run_evaluation.episode_handler("sonnet")
    result = asyncio.run(handler({"model": {"name": "x"}}, inline["input"], tools, inline["variables"]))

    assert len(calls) == len(row["metadata"]["test_entry"]["question"])
    assert result["usage"]["total"] == 10 * len(calls)
    assert score_with_bfcl(row, result["output"], model_name="claude-sonnet-5-5") == 1.0


class FakeClient:
    def __init__(self):
        self.events = []

    def track(self, key, context, data=None, value=None):
        # The real ldclient raises on a plain dict; mirror that.
        assert isinstance(context, Context), "track() needs an ldclient Context"
        assert context.valid and context.kind == "request"
        self.events.append((key, value))


def experiment_harness(monkeypatch, make_policy):
    client = FakeClient()
    first_turns = []
    captured = {}

    async def fake_inspect(key, context):
        return {
            "config": {
                "model": {"name": "claude-sonnet-5-5"},
                "tools": {key: {"description": "", "parameters": {}} for key in TOOL_KEYS},
            },
            "meta": {"variationKey": "claude-sonnet-5-5"},
        }

    class FakeInstance:
        def __init__(self, tool_handlers):
            self.policy = make_policy(tool_handlers)

        async def invoke(self, user_input, context, variables=None, history=None):
            if not history:
                first_turns.append(user_input)
            return SimpleNamespace(response=await self.policy(user_input, history))

    def fake_config(key, handler, tool_handlers):
        captured["tool_keys"] = sorted(tool_handlers)
        return FakeInstance(tool_handlers)

    monkeypatch.setattr(run_experiment, "inspect_config", fake_inspect)
    monkeypatch.setattr(run_experiment, "config", fake_config)
    monkeypatch.setattr(run_experiment, "get_client", lambda: client)
    monkeypatch.setattr(run_experiment, "provider_handlers", lambda config_key: [])
    real_inspect = fake_inspect

    async def recording_inspect(key, context):
        captured["config_key"], captured["context_key"] = key, context["key"]
        return await real_inspect(key, context)

    monkeypatch.setattr(run_experiment, "inspect_config", recording_inspect)
    return client, first_turns, captured


def test_experiment_trial_resolves_and_emits_both_events(monkeypatch):
    row = rows("bfcl_pilot.jsonl")[0]
    client, _, captured = experiment_harness(monkeypatch, lambda tools: ground_truth_turns(row, tools))
    log = io.StringIO()

    audit = asyncio.run(run_experiment.run_trial(row, output_log=log))

    assert captured["tool_keys"] == sorted(TOOL_KEYS)
    assert audit["resolved"] is True
    assert audit["attempts"] == 1
    assert [key for key, _ in client.events] == [
        "task-resolved",
        "task-resolution-duration-ms",
        "task-resolved-within-slo",
    ]
    assert client.events[1][1] == pytest.approx(audit["duration_ms"], abs=0.01)
    assert audit["within_slo"] is True
    assert json.loads(log.getvalue())["source_id"] == row["metadata"]["source_id"]


def test_experiment_trial_failure_repairs_once_and_emits_nothing(monkeypatch):
    row = rows("bfcl_pilot.jsonl")[1]

    def silent(tools):
        async def call_turn(user_text, history):
            return "I cannot help with that."

        return call_turn

    client, first_turns, _ = experiment_harness(monkeypatch, silent)

    audit = asyncio.run(run_experiment.run_trial(row, output_log=io.StringIO()))

    assert audit["resolved"] is False
    assert audit["attempts"] == 2
    assert audit["duration_ms"] is None
    assert audit["validator_error_type"] == "multi_turn:empty_turn_model_response"
    assert client.events == []
    assert not first_turns[0].startswith("The official BFCL evaluator rejected")
    assert first_turns[1].startswith("The official BFCL evaluator rejected")


def test_tool_search_trial_uses_its_own_config_and_request_keys(monkeypatch):
    row = rows("bfcl_pilot.jsonl")[0]
    client, _, captured = experiment_harness(monkeypatch, lambda tools: ground_truth_turns(row, tools))

    audit = asyncio.run(
        run_experiment.run_trial(row, output_log=io.StringIO(), config_key="bfcl-tool-search")
    )
    assert [key for key, _ in client.events][:2] == ["task-resolved", "task-resolution-duration-ms"]

    source_id = row["metadata"]["source_id"]
    assert captured["config_key"] == "bfcl-tool-search"
    assert captured["context_key"] == f"bfcl-tool-search-{source_id}"
    assert audit["request_id"] == f"bfcl-tool-search-{source_id}"
    assert audit["tool_searches"] == 0
    assert audit["resolved"] is True


def test_model_comparison_keeps_its_original_request_keys():
    assert run_experiment.request_key("bfcl-model-comparison", "multi_turn_base_1") == "bfcl-multi_turn_base_1"


def test_only_the_tool_search_experiment_gets_the_tool_search_handler(monkeypatch):
    monkeypatch.setattr(run_experiment, "create_tool_search_handler", lambda: "tool-search")
    monkeypatch.setattr(run_experiment, "create_claude_messages_handler", lambda: "stock")
    assert run_experiment.provider_handlers("bfcl-model-comparison") == ["stock"]
    assert run_experiment.provider_handlers("bfcl-tool-search") == ["tool-search"]


def test_tool_search_arms_score_independently_on_the_same_task():
    """Both arms share claude-sonnet-5-5. Under one BFCL model_name, a partly wrong first
    arm leaves checker state behind and a correct second arm scores 0 (15 of 30 pilot
    tasks before the fix)."""
    row = rows("bfcl_pilot.jsonl")[0]
    tools = run_evaluation.tool_implementations(TOOL_KEYS)
    episode = asyncio.run(
        run_evaluation.run_episode(
            row["metadata"]["test_entry"], ground_truth_turns(row, tools), scope="arm-check"
        )
    )
    good = json.loads(episode.as_output())
    turns = good["turns"]
    partly_wrong = {**good, "turns": [turns[0]] + [[["ls(a=True)"]] for _ in turns[1:]]}
    first, second = sorted(run_evaluation.ARMS["bfcl-tool-search"])

    assert run_evaluation.scorer_for(first).fn(row, json.dumps(partly_wrong)) == 0.0
    assert run_evaluation.scorer_for(second).fn(row, json.dumps(good)) == 1.0


def test_search_tally_is_reset_when_a_trial_raises(monkeypatch):
    from llm_p90.tool_search_handler import SEARCH_TALLY

    row = rows("bfcl_pilot.jsonl")[0]

    def failing(tools):
        async def call_turn(user_text, history):
            raise ConnectionError("provider down")

        return call_turn

    experiment_harness(monkeypatch, failing)

    async def trial_then_check():
        with pytest.raises(ConnectionError):
            await run_experiment.run_trial(row, output_log=io.StringIO())
        return SEARCH_TALLY.get()

    assert asyncio.run(trial_then_check()) is None


def test_resolved_task_over_the_slo_sends_no_slo_event(monkeypatch):
    row = rows("bfcl_pilot.jsonl")[0]
    client, _, _ = experiment_harness(monkeypatch, lambda tools: ground_truth_turns(row, tools))
    monkeypatch.setattr(run_experiment, "SLO_MS", -1)

    audit = asyncio.run(run_experiment.run_trial(row, output_log=io.StringIO()))

    assert audit["resolved"] is True
    assert audit["within_slo"] is False
    assert [key for key, _ in client.events] == ["task-resolved", "task-resolution-duration-ms"]


def test_run_id_gives_a_rerun_fresh_request_keys():
    assert run_experiment.request_key("bfcl-tool-search", "multi_turn_base_1") == "bfcl-tool-search-multi_turn_base_1"
    assert run_experiment.request_key("bfcl-tool-search", "multi_turn_base_1", "r2") == "bfcl-tool-search-r2-multi_turn_base_1"
    assert run_experiment.request_key("bfcl-model-comparison", "multi_turn_base_1", "r2") == "bfcl-model-comparison-r2-multi_turn_base_1"


def test_select_rows_resumes_at_a_zero_based_offset():
    sample = [{"id": value} for value in range(5)]

    assert run_experiment.select_rows(sample, start=2, tasks=2) == [
        {"id": 2},
        {"id": 3},
    ]


def test_select_rows_rejects_an_out_of_range_window():
    with pytest.raises(ValueError, match="plus --tasks"):
        run_experiment.select_rows([{"id": 0}], start=1, tasks=1)


@pytest.mark.parametrize("tasks", [0, -1])
def test_select_rows_rejects_non_positive_task_counts(tasks):
    with pytest.raises(ValueError, match="--tasks must be one or greater"):
        run_experiment.select_rows([{"id": 0}], start=0, tasks=tasks)
