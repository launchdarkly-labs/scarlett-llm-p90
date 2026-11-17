import asyncio
import json
import re

import pytest
from bfcl_eval.constants.enums import ModelStyle
from bfcl_eval.constants.eval_config import MULTI_TURN_FUNC_DOC_PATH
from bfcl_eval.constants.executable_backend_config import (
    MULTI_TURN_FUNC_DOC_FILE_MAPPING,
)
from bfcl_eval.constants.type_mappings import GORILLA_TO_OPENAPI
from bfcl_eval.model_handler.utils import convert_to_tool
from bfcl_policies import ROOT, TOOL_KEYS, ground_truth_turns, rows

from llm_p90.bfcl_loop import (
    STEP_LIMIT_MESSAGE,
    repair_note_from,
    run_episode,
    tool_implementations,
    tool_key,
)
from llm_p90.scoring import evaluate_with_bfcl

TOOLS = tool_implementations(TOOL_KEYS)


def test_committed_tool_definitions_are_bfcls_own_conversion():
    committed = json.loads((ROOT / "tools" / "bfcl_tools.json").read_text())
    docs = []
    for cls in committed["classes"]:
        path = MULTI_TURN_FUNC_DOC_PATH / MULTI_TURN_FUNC_DOC_FILE_MAPPING[cls]
        docs += [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    converted = convert_to_tool(docs, GORILLA_TO_OPENAPI, ModelStyle.ANTHROPIC)
    trimmed = set(committed["trimmed_response_schema"])

    assert len(committed["tools"]) == len(converted) == 128
    assert trimmed == {"book_flight", "displayCarStatus"}
    for ours, bfcl in zip(committed["tools"], converted, strict=True):
        assert ours["key"] == tool_key(bfcl["name"]) == ours["key"].lower()
        assert ours["function"] == bfcl["name"]
        assert ours["schema"] == bfcl["input_schema"]
        assert len(ours["description"]) <= committed["description_limit"]
        if ours["function"] in trimmed:
            # Only BFCL's appended response-schema text is dropped.
            assert bfcl["description"].startswith(ours["description"])
            assert bfcl["description"][len(ours["description"]):].startswith(
                " The response field has the following schema:"
            )
        else:
            assert ours["description"] == bfcl["description"]


@pytest.mark.parametrize("split", ["bfcl_pilot.jsonl", "bfcl_confirmatory.jsonl"])
def test_every_function_the_ground_truth_uses_is_an_attached_tool(split):
    for row in rows(split):
        used = {tool_key(name) for name in re.findall(r"([A-Za-z_]\w*)\(", row["expected_output"])}
        assert used <= set(TOOL_KEYS), row["metadata"]["source_id"]


@pytest.mark.parametrize("split", ["bfcl_pilot.jsonl", "bfcl_confirmatory.jsonl"])
def test_every_task_resolves_when_the_model_plays_the_ground_truth(split):
    for row in rows(split):
        entry = row["metadata"]["test_entry"]
        episode = asyncio.run(run_episode(entry, ground_truth_turns(row, TOOLS), scope="gt"))

        result = evaluate_with_bfcl(row, episode.as_output(), model_name="gt")

        assert result["valid"] is True, (entry["id"], result)


def test_a_model_that_never_calls_tools_fails_the_official_checker():
    row = rows("bfcl_pilot.jsonl")[0]

    async def silent(user_text, history):
        return "I cannot help with that."

    episode = asyncio.run(run_episode(row["metadata"]["test_entry"], silent, scope="silent"))
    result = evaluate_with_bfcl(row, episode.as_output(), model_name="silent")

    assert result["valid"] is False
    assert result["error_type"] == "multi_turn:empty_turn_model_response"


def test_tool_results_come_from_bfcls_backend():
    row = rows("bfcl_pilot.jsonl")[0]
    policy = ground_truth_turns(row, TOOLS)

    asyncio.run(run_episode(row["metadata"]["test_entry"], policy, scope="results"))

    # Row 0's first call is get_stock_info; the backend answers with the live price.
    assert "457.23" in policy.seen_results[0]


def test_a_tool_outside_the_tasks_backends_returns_an_error_instead_of_raising():
    row = rows("bfcl_pilot.jsonl")[0]  # TwitterAPI and TradingBot only
    returned = []

    async def strays(user_text, history):
        returned.append(TOOLS["lock_doors"]({"unlock": False, "door": ["driver"]}))
        return "Done."

    asyncio.run(run_episode(row["metadata"]["test_entry"], strays, scope="stray"))

    assert returned[0].startswith("Error during execution")


def test_history_carries_earlier_calls_and_results_as_text():
    row = rows("bfcl_pilot.jsonl")[0]
    policy = ground_truth_turns(row, TOOLS)
    histories = []

    async def recording(user_text, history):
        histories.append(history)
        return await policy(user_text, history)

    asyncio.run(run_episode(row["metadata"]["test_entry"], recording, scope="history"))

    second_turn = histories[1]
    assert second_turn[0]["role"] == "user"
    assert "get_stock_info(symbol='OMEG') ->" in second_turn[1]["content"]


def test_repair_attempt_starts_from_the_initial_state_with_checker_feedback_only():
    row = rows("bfcl_pilot.jsonl")[0]
    entry = row["metadata"]["test_entry"]

    async def wrong(user_text, history):
        TOOLS["post_tweet"]({"content": "wrong", "tags": [], "mentions": []})
        return "Done."

    first = asyncio.run(run_episode(entry, wrong, scope="repair-a1"))
    validation = evaluate_with_bfcl(row, first.as_output(), model_name="repair-a1")
    assert validation["valid"] is False

    note = repair_note_from(validation)
    first_turns = []
    policy = ground_truth_turns(row, TOOLS)

    async def recording(user_text, history):
        first_turns.append(user_text)
        return await policy(user_text, history)

    second = asyncio.run(run_episode(entry, recording, scope="repair-a2", repair_note=note))

    assert first_turns[0].startswith("The official BFCL evaluator rejected")
    assert not first_turns[1].startswith("The official BFCL evaluator rejected")
    assert "skyrocketing at $457.23" not in note
    assert evaluate_with_bfcl(row, second.as_output(), model_name="repair-a2")["valid"]


def test_hitting_the_handlers_step_limit_fails_the_task_as_force_terminated():
    row = rows("bfcl_pilot.jsonl")[0]

    async def looping(user_text, history):
        TOOLS["get_stock_info"]({"symbol": "OMEG"})
        raise RuntimeError(f"{STEP_LIMIT_MESSAGE} (10)")

    episode = asyncio.run(run_episode(row["metadata"]["test_entry"], looping, scope="loop"))
    result = evaluate_with_bfcl(row, episode.as_output(), model_name="loop")

    assert episode.force_quit is True
    assert result["valid"] is False
    assert result["error_type"] == "multi_turn:force_terminated"


def test_renamed_tool_keys_run_the_original_bfcl_function():
    row = next(r for r in rows("bfcl_pilot.jsonl") if "VehicleControlAPI" in r["metadata"]["test_entry"]["involved_classes"])
    calls = []

    async def lock(user_text, history):
        calls.append(TOOLS["lock_doors"]({"unlock": False, "door": ["driver"]}))
        return "Done."

    episode = asyncio.run(run_episode(row["metadata"]["test_entry"], lock, scope="renamed"))

    assert not calls[0].startswith("Error during execution")
    assert episode.decoded_turns[0][0][0].startswith("lockDoors(")


def test_other_runtime_errors_are_not_swallowed():
    row = rows("bfcl_pilot.jsonl")[0]

    async def broken(user_text, history):
        raise RuntimeError("provider outage")

    with pytest.raises(RuntimeError, match="provider outage"):
        asyncio.run(run_episode(row["metadata"]["test_entry"], broken, scope="broken"))
