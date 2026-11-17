"""Export BFCL's multi-turn functions as Agent Management tool definitions.

Every function from the backends the 200 tasks use, converted to JSON Schema
by BFCL's own ``convert_to_tool`` (Anthropic style), written to
tools/bfcl_tools.json.

Two adjustments. Tool keys are lowercase, because the AI SDK's evaluations
reject uppercase keys: the ten camelCase vehicle functions become snake_case
(lockDoors becomes lock_doors), and "function" records the BFCL name each key
runs. And Agent Management caps a tool description at 1024 characters. BFCL
appends each function's response schema to its description, and for two
functions that pushes it over the cap. For those only, the appended
"The response field has the following schema: ..." text is dropped. The model
still sees every actual result when the call executes.

    uv run python scripts/export_bfcl_tools.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from bfcl_eval.constants.enums import ModelStyle
from bfcl_eval.constants.eval_config import MULTI_TURN_FUNC_DOC_PATH
from bfcl_eval.constants.executable_backend_config import (
    MULTI_TURN_FUNC_DOC_FILE_MAPPING,
)
from bfcl_eval.constants.type_mappings import GORILLA_TO_OPENAPI
from bfcl_eval.model_handler.utils import convert_to_tool

ROOT = Path(__file__).parents[1]
DESCRIPTION_LIMIT = 1024
RESPONSE_SCHEMA_MARKER = " The response field has the following schema:"


def tool_key(function_name: str) -> str:
    # Same rule as llm_p90.bfcl_loop.tool_key; the tests check they agree.
    return re.sub(r"(?<!^)(?=[A-Z])", "_", function_name).lower()


def task_classes() -> list[str]:
    classes = set()
    for name in ("bfcl_pilot.jsonl", "bfcl_confirmatory.jsonl"):
        for line in (ROOT / "data" / name).read_text().splitlines():
            if line.strip():
                classes.update(json.loads(line)["metadata"]["test_entry"]["involved_classes"])
    return sorted(classes)


def fit_description(description: str) -> str:
    if len(description) <= DESCRIPTION_LIMIT:
        return description
    cut = description.find(RESPONSE_SCHEMA_MARKER)
    if cut <= 0 or cut > DESCRIPTION_LIMIT:
        raise ValueError(f"cannot fit description within {DESCRIPTION_LIMIT} characters")
    return description[:cut]


def tool_definitions() -> dict[str, object]:
    classes = task_classes()
    docs = []
    for cls in classes:
        path = MULTI_TURN_FUNC_DOC_PATH / MULTI_TURN_FUNC_DOC_FILE_MAPPING[cls]
        docs += [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    tools = convert_to_tool(docs, GORILLA_TO_OPENAPI, ModelStyle.ANTHROPIC)
    return {
        "source": "bfcl_eval convert_to_tool(..., GORILLA_TO_OPENAPI, ModelStyle.ANTHROPIC)",
        "description_limit": DESCRIPTION_LIMIT,
        "trimmed_response_schema": sorted(
            tool["name"] for tool in tools if len(tool["description"]) > DESCRIPTION_LIMIT
        ),
        "classes": classes,
        "tools": [
            {
                "key": tool_key(tool["name"]),
                "function": tool["name"],
                "description": fit_description(tool["description"]),
                "schema": tool["input_schema"],
            }
            for tool in tools
        ],
    }


def main() -> None:
    out = ROOT / "tools" / "bfcl_tools.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    definitions = tool_definitions()
    out.write_text(json.dumps(definitions, indent=2) + "\n")
    print(f"wrote {len(definitions['tools'])} tools; trimmed {definitions['trimmed_response_schema']}")


if __name__ == "__main__":
    main()
