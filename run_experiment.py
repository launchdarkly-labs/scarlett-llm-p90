"""Run the paid, request-randomized confirmatory experiment.

LaunchDarkly owns assignment and inference. This runner owns the BFCL workload,
run interactively turn by turn as BFCL runs it (see llm_p90.bfcl_loop), one
repair attempt, complete-task timing, and three custom events:

* task-resolved, emitted only after BFCL accepts the task
* task-resolution-duration-ms, emitted only for resolved tasks
* task-resolved-within-slo, emitted only for tasks resolved within SLO_MS. The share
  of requests that send it exists even when too many tasks fail for a p90 to exist.

Token usage needs no custom event: the AI SDK tracks $ld:ai:tokens:* after every call.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from launchdarkly_ai_claude_messages import create_claude_messages_handler
from launchdarkly_ai_server import config, get_client, inspect_config, shutdown
from ldclient import Context

from llm_p90.bfcl_loop import repair_note_from, run_episode, tool_implementations
from llm_p90.protocol import TrialState
from llm_p90.scoring import evaluate_with_bfcl
from llm_p90.tool_search_handler import (
    SEARCH_TALLY,
    create_tool_search_handler,
    tool_search_mode,
)

CONFIG_KEY = "bfcl-model-comparison"
TOOL_SEARCH_CONFIG_KEY = "bfcl-tool-search"
RESOLUTION_EVENT = "task-resolved"
DURATION_EVENT = "task-resolution-duration-ms"
SLO_EVENT = "task-resolved-within-slo"
#: Fixed before the tool-search experiment's first paid call (see protocol.json).
SLO_MS = 30_000


def read_jsonl(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def select_rows(
    rows: list[dict[str, object]], *, start: int, tasks: int
) -> list[dict[str, object]]:
    if start < 0:
        raise ValueError("--start must be zero or greater")
    if tasks < 1:
        raise ValueError("--tasks must be one or greater")
    if start + tasks > len(rows):
        raise ValueError(
            f"--start {start} plus --tasks {tasks} exceeds the "
            f"{len(rows)} available rows"
        )
    return rows[start : start + tasks]


def provider_handlers(config_key: str = CONFIG_KEY) -> list[object]:
    """The stock Claude handler, except for the tool-search experiment, whose arms need
    ``defer_loading`` (see llm_p90.tool_search_handler). Both of its arms use the same
    handler, so they differ only by the variation's ``model.parameters.tool_search``."""
    if config_key == TOOL_SEARCH_CONFIG_KEY:
        return [create_tool_search_handler()]
    return [create_claude_messages_handler()]


def request_key(config_key: str, source_id: str, run_id: str = "") -> str:
    """One request context per task per experiment, so events never cross experiments.
    ``run_id`` gives a rerun fresh context keys, so LaunchDarkly assigns it afresh."""
    prefix = f"{config_key}-{run_id}" if run_id else config_key
    if config_key == CONFIG_KEY and not run_id:
        return f"bfcl-{source_id}"
    return f"{prefix}-{source_id}"


def tool_keys(inspected: dict[str, object]) -> list[str]:
    """Tool keys attached to the served variation. Definitions stay in Agent Management."""
    current = inspected.get("config")
    if not isinstance(current, dict):
        raise TypeError(
            "Agent Management returned no usable config: it is unreachable, disabled, or "
            "fails SDK validation (a variation needs at least one message)"
        )
    tools = current.get("tools")
    if not isinstance(tools, dict) or not tools:
        raise TypeError("Agent Management variation has no tools attached")
    return sorted(tools)


def model_name(inspected: dict[str, object]) -> str:
    current = inspected.get("config")
    if not isinstance(current, dict):
        raise TypeError("Agent Management returned no usable config")
    model = current.get("model")
    if not isinstance(model, dict) or not isinstance(model.get("name"), str):
        raise TypeError("Agent Management variation has no model name")
    return model["name"]


def variation_key(inspected: dict[str, object]) -> str:
    meta = inspected.get("meta")
    if isinstance(meta, dict):
        value = meta.get("variationKey") or meta.get("variation_key")
        if isinstance(value, str):
            return value
    return "unknown"


async def run_trial(
    row: dict[str, object],
    *,
    output_log: object,
    config_key: str = CONFIG_KEY,
    handlers: list[object] | None = None,
    run_id: str = "",
) -> dict[str, object]:
    metadata = row.get("metadata")
    if not isinstance(metadata, dict) or not isinstance(metadata.get("source_id"), str):
        raise TypeError("dataset row requires metadata.source_id")
    source_id = metadata["source_id"]
    context = {
        "kind": "request",
        "key": request_key(config_key, source_id, run_id),
        "sourceId": source_id,
    }
    inspected = await inspect_config(config_key, context)
    chosen_model = model_name(inspected)
    chosen_variation = variation_key(inspected)
    test_entry = metadata["test_entry"]
    instance = config(
        key=config_key,
        handler=handlers if handlers is not None else provider_handlers(config_key),
        tool_handlers=tool_implementations(tool_keys(inspected)),
    )

    async def call_turn(user_text: str, history: list[dict[str, str]]) -> str:
        response = await instance.invoke(user_text, context, history=history)
        return str(response.response)

    tally = {"searches": 0}
    tally_token = SEARCH_TALLY.set(tally)
    state = TrialState(max_attempts=2)
    repair_note: str | None = None
    validation: dict[str, object] = {"valid": False}
    started_ns = time.perf_counter_ns()

    try:
        while state.attempts_started < state.max_attempts:
            attempt = state.begin_attempt()
            # A distinct backend scope per attempt, so a repair starts from the
            # task's initial state rather than the first attempt's side effects.
            scope = f"{chosen_model}-attempt{attempt}"
            episode = await run_episode(
                test_entry, call_turn, scope=scope, repair_note=repair_note
            )
            validation = dict(evaluate_with_bfcl(row, episode.as_output(), model_name=scope))
            if validation.get("valid") is True:
                break
            repair_note = repair_note_from(validation)
        elapsed_ms = (time.perf_counter_ns() - started_ns) / 1_000_000
    finally:
        SEARCH_TALLY.reset(tally_token)
    resolved = validation.get("valid") is True
    client = get_client()
    if resolved:
        # The AI SDK takes a dict context; the base SDK's track() needs a Context.
        ld_context = Context.from_dict(context)
        client.track(RESOLUTION_EVENT, ld_context)
        client.track(DURATION_EVENT, ld_context, None, elapsed_ms)
        if elapsed_ms <= SLO_MS:
            client.track(SLO_EVENT, ld_context)

    audit_row = {
        "source_id": source_id,
        "request_id": context["key"],
        "variation": chosen_variation,
        "model": chosen_model,
        "attempts": state.attempts_started,
        "tool_search": tool_search_mode(inspected["config"]),
        "tool_searches": tally["searches"],
        "resolved": resolved,
        "duration_ms": round(elapsed_ms, 3) if resolved else None,
        "within_slo": resolved and elapsed_ms <= SLO_MS,
        "validator_error_type": None if resolved else validation.get("error_type"),
        "validator_error": None
        if resolved
        else validation.get("error_message") or validation.get("error"),
    }
    output_log.write(json.dumps(audit_row, separators=(",", ":")) + "\n")
    output_log.flush()
    print(
        f"{source_id}: variation={chosen_variation} attempts={state.attempts_started} "
        f"resolved={resolved}"
    )
    return audit_row


async def async_main(args: argparse.Namespace) -> None:
    rows = read_jsonl(args.dataset)
    selected = select_rows(rows, start=args.start, tasks=args.tasks)
    if args.output.exists() and args.output.stat().st_size > 0:
        raise ValueError(
            f"{args.output} already has rows; pass a new --output for each run"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        handlers = provider_handlers(args.config)
        with args.output.open("w", encoding="utf-8") as output_log:
            for row in selected:
                await run_trial(
                    row,
                    output_log=output_log,
                    config_key=args.config,
                    handlers=handlers,
                    run_id=args.run_id,
                )
    finally:
        await shutdown()


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/bfcl_confirmatory.jsonl"),
    )
    parser.add_argument("--tasks", type=int, required=True)
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Zero-based row offset. Use it to resume at the first row not written.",
    )
    parser.add_argument(
        "--config",
        choices=[CONFIG_KEY, TOOL_SEARCH_CONFIG_KEY],
        default=CONFIG_KEY,
        help="Agent Management config, which is also the experiment, to run tasks through.",
    )
    parser.add_argument(
        "--run-id",
        default="",
        help="Label for a rerun. Gives every task a fresh request context, so a new "
        "iteration randomizes again instead of repeating the earlier assignment.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("logs/confirmatory.jsonl"),
    )
    parser.add_argument(
        "--confirm-paid-run",
        action="store_true",
        help="Required acknowledgement because this command spends provider API budget.",
    )
    args = parser.parse_args()
    if not args.confirm_paid_run:
        parser.error("--confirm-paid-run is required")
    if not os.getenv("LD_SDK_KEY"):
        parser.error("LD_SDK_KEY is required")
    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
