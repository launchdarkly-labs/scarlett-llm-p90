"""Scripted models for offline tests. They call tools the way a provider's tool loop does."""

import ast
import json
from pathlib import Path

from llm_p90.bfcl_loop import function_docs, tool_key

ROOT = Path(__file__).parents[1]
TOOL_KEYS = [tool["key"] for tool in json.loads((ROOT / "tools" / "bfcl_tools.json").read_text())["tools"]]


def rows(name):
    return [
        json.loads(line)
        for line in (ROOT / "data" / name).read_text().splitlines()
        if line
    ]


def ground_truth_arguments(call, docs_by_name):
    """Parse a ground-truth call into (name, arguments), naming positional arguments."""
    node = ast.parse(call, mode="eval").body
    name = node.func.id
    params = list(docs_by_name[name]["parameters"]["properties"])
    arguments = {params[i]: ast.literal_eval(arg) for i, arg in enumerate(node.args)}
    arguments.update({kw.arg: ast.literal_eval(kw.value) for kw in node.keywords})
    return name, arguments


def ground_truth_turns(row, tools):
    """A call_turn that plays each turn's ground-truth calls through ``tools``."""
    docs = {doc["name"]: doc for doc in function_docs(row["metadata"]["test_entry"])}
    turns = iter(json.loads(row["expected_output"]))
    seen_results = []

    async def call_turn(user_text, history):
        for call in next(turns):
            name, arguments = ground_truth_arguments(call, docs)
            seen_results.append(tools[tool_key(name)](arguments))
        return "Done."

    call_turn.seen_results = seen_results
    return call_turn
