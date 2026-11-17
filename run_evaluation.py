"""Run the paired, first-attempt pilot with Agent Management evaluations from code.

Each dataset row is a full BFCL multi-turn task. The tool definitions come
from the Agent Management tool library, attached to each variation; the Claude
handler sends them through the provider's native tool parameter. A wrapper
handler plays the task turn by turn (see llm_p90.bfcl_loop), and every tool
call executes on BFCL's simulated backends. The official BFCL checker then
scores the calls.

This calibration run does not estimate the confirmatory two-attempt resolution
rate. The confirmatory runner owns the repair policy and complete-task timing.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from launchdarkly_ai_claude_messages import create_claude_messages_handler
from launchdarkly_ai_server import (
    AIConfig,
    Scorer,
    create_handler,
    init_evaluations,
    inspect_config,
    shutdown,
)

from llm_p90.bfcl_loop import run_episode, tool_implementations
from llm_p90.scoring import score_with_bfcl
from llm_p90.tool_search_handler import create_tool_search_handler

CONFIG_KEY = "bfcl-model-comparison"
TOOL_SEARCH_CONFIG_KEY = "bfcl-tool-search"
ENTRY_VARIABLE = "bfcl_entry"

#: The variations each config's pilot runs, keyed by the short name used on the command line.
ARMS = {
    CONFIG_KEY: {
        "haiku": {"model": "claude-haiku-5-5", "variation": "claude-haiku-5-5"},
        "sonnet": {"model": "claude-sonnet-5-5", "variation": "claude-sonnet-5-5"},
    },
    TOOL_SEARCH_CONFIG_KEY: {
        "all-tools": {"model": "claude-haiku-5-5", "variation": "all-tools"},
        "tool-search": {"model": "claude-haiku-5-5", "variation": "tool-search"},
    },
}


def required(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def provider_handler(config_key: str = CONFIG_KEY) -> Any:
    if config_key == TOOL_SEARCH_CONFIG_KEY:
        return create_tool_search_handler()
    return create_claude_messages_handler()


def episode_handler(model_key: str, config_key: str = CONFIG_KEY) -> Any:
    """Wrap the provider handler so one evaluation row runs one whole BFCL episode."""
    inner = provider_handler(config_key)

    async def run_task(
        config: dict[str, Any],
        user_input: str | None,
        tools: Any,
        variables: dict[str, Any] | None,
        history: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        test_entry = json.loads(dict(variables or {})[ENTRY_VARIABLE])
        usage: dict[str, float] = {}

        async def call_turn(user_text: str, turns: list[dict[str, str]]) -> str:
            result = await inner(config, user_text, tools, {}, turns)
            for key, value in (result.get("usage") or {}).items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    usage[key] = usage.get(key, 0) + value
            return str(result.get("output", ""))

        episode = await run_episode(test_entry, call_turn, scope=f"pilot-{model_key}")
        return {"output": episode.as_output(), "usage": usage}

    return create_handler(inner.provides_for, run_task, providers=inner.providers)


def load_inline_dataset(path: Path) -> list[dict[str, object]]:
    rows = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if "expected_output" in row:
            row["expectedOutput"] = row.pop("expected_output")
        test_entry = row["metadata"]["test_entry"]
        row["variables"] = {ENTRY_VARIABLE: json.dumps(test_entry, separators=(",", ":"))}
        rows.append(row)
    if not rows:
        raise ValueError(f"{path} contains no dataset rows")
    return rows


async def attached_tool_keys(config_key: str = CONFIG_KEY) -> list[str]:
    """Read the tool keys attached to the config from Agent Management."""
    inspected = await inspect_config(
        config_key, {"kind": "request", "key": f"{config_key}-pilot-tool-discovery"}
    )
    current = inspected.get("config")
    if not isinstance(current, dict):
        raise TypeError(
            f"Agent Management returned no usable config for {config_key}: it is "
            "unreachable, disabled, or fails SDK validation (a variation needs at "
            "least one message)"
        )
    tools = current.get("tools")
    if not isinstance(tools, dict) or not tools:
        raise RuntimeError(f"{config_key} has no tools attached")
    return sorted(tools)


def scorer_for(model_key: str) -> Scorer:
    """The official BFCL check for one arm. BFCL caches checker state per model_name and
    task, and both tool-search arms share a model, so each arm scores under its own name."""
    return Scorer(
        name=f"official-bfcl-{model_key}",
        fn=partial(score_with_bfcl, model_name=f"pilot-{model_key}"),
        threshold=1.0,
    )


async def run_model(
    model_key: str,
    dataset: list[dict[str, object]],
    tool_keys: list[str],
    concurrency: int,
    config_key: str = CONFIG_KEY,
) -> object:
    model = ARMS[config_key][model_key]
    evals = init_evaluations(
        project_key=required("LD_PROJECT_KEY"),
        api_key=required("LD_OPERATOR_TOKEN"),
        sdk_key=required("LD_SDK_KEY"),
    )
    implementations = tool_implementations(tool_keys)
    tools = [
        await evals.tools.get(key, implementation=implementations[key])
        for key in tool_keys
    ]
    criteria = [scorer_for(model_key)]
    unique = datetime.now(UTC).strftime("%Y%m%d-%H%M%S-%f")
    return await evals.run(
        key=f"bfcl-pilot-{model_key}-{unique}",
        dataset=dataset,
        ai_config=AIConfig(key=config_key, variation=model["variation"]),
        handlers=[episode_handler(model_key, config_key)],
        tools=tools,
        criteria=criteria,
        concurrency=concurrency,
    )


async def async_main(args: argparse.Namespace) -> None:
    dataset = load_inline_dataset(args.dataset)
    if args.tasks is not None:
        dataset = dataset[: args.tasks]
    try:
        tool_keys = await attached_tool_keys(args.config)
        for model_key in args.models or sorted(ARMS[args.config]):
            if model_key not in ARMS[args.config]:
                raise ValueError(f"{model_key} is not an arm of {args.config}")
            result = await run_model(
                model_key, dataset, tool_keys, args.concurrency, args.config
            )
            print(result)
    finally:
        await shutdown()


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/bfcl_pilot.jsonl"),
    )
    parser.add_argument(
        "--config",
        choices=sorted(ARMS),
        default=CONFIG_KEY,
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Arms to run (default: every arm of --config).",
    )
    parser.add_argument(
        "--tasks",
        type=int,
        default=None,
        help="Run only the first N rows (default: all 30).",
    )
    parser.add_argument("--concurrency", type=int, default=1)
    args = parser.parse_args()
    if args.concurrency != 1:
        raise ValueError("protocol.json pre-registers concurrency=1")
    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
