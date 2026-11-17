"""Create the bfcl-tool-search config: two Haiku 5.5 variations that differ only in
the ``tool_search`` model parameter.

Use this when your MCP server has no create-agent-config tool. It copies the system
message, parameters, and all 128 tool references from bfcl-model-comparison's Haiku
variation, so the only difference between the arms is deferred tool loading. It skips
anything that already exists, then reads both variations back and checks them.
"""

from __future__ import annotations

import os
import sys

from create_bfcl_tools import http_with_token, with_rate_limit_retries
from dotenv import load_dotenv

SOURCE_CONFIG = "bfcl-model-comparison"
SOURCE_VARIATION = "claude-haiku-5-5"
CONFIG_KEY = "bfcl-tool-search"
#: Extra model parameters per arm. ``tool_search`` is a parameter, not ``model.custom``,
#: because evaluations from code drop ``model.custom`` (see llm_p90.tool_search_handler).
ARMS = {
    "all-tools": {"name": "Haiku 5.5, all tools up front", "parameters": {}},
    "tool-search": {"name": "Haiku 5.5, tool search", "parameters": {"tool_search": "bm25"}},
}


def arm_model(template: dict, arm: dict) -> dict:
    return {
        "modelName": template["model"]["modelName"],
        "parameters": {**template["model"]["parameters"], **arm["parameters"]},
        "custom": {},
    }


def message(body: object) -> str:
    return body.get("message", "") if isinstance(body, dict) else str(body)


def main() -> None:
    load_dotenv()
    call = with_rate_limit_retries(http_with_token(os.environ["LD_OPERATOR_TOKEN"], beta=True))
    project = os.environ.get("LD_PROJECT_KEY", "llm-p90-experiment")
    base = f"/projects/{project}/ai-configs"

    status, source = call("GET", f"{base}/{SOURCE_CONFIG}", None)
    if status != 200:
        sys.exit(f"could not read {SOURCE_CONFIG}: HTTP {status} {message(source)}")
    template = next((v for v in source["variations"] if v["key"] == SOURCE_VARIATION), None)
    if template is None:
        found = ", ".join(v["key"] for v in source["variations"])
        sys.exit(f"variation {SOURCE_VARIATION} not found in {SOURCE_CONFIG} (found: {found})")

    # Check what exists first: re-posting an existing config returns a misleading 400,
    # not a 409, so the script cannot rely on conflict codes to be rerunnable.
    status, current = call("GET", f"{base}/{CONFIG_KEY}", None)
    if status == 200:
        print(f"config {CONFIG_KEY}: already exists")
        existing = {v["key"] for v in current.get("variations", [])}
    elif status == 404:
        status, body = call("POST", base, {
            "key": CONFIG_KEY,
            "name": "BFCL tool search",
            "mode": "completion",
            "description": "Claude Haiku 5.5 with all 128 BFCL tools up front versus behind "
            "Anthropic tool search. Same model, prompt, parameters, and tools.",
            "tags": ["devrel", "tutorial", "bfcl"],
        })
        print(f"config {CONFIG_KEY}: HTTP {status}")
        if status not in (200, 201):
            sys.exit(f"creating {CONFIG_KEY} failed: {message(body)}")
        existing = set()
    else:
        sys.exit(f"could not read {CONFIG_KEY}: HTTP {status} {message(current)}")

    current_models = {
        v["key"]: (v.get("model") or {}, v.get("modelConfigKey"))
        for v in (current.get("variations", []) if existing else [])
    }
    for key, arm in ARMS.items():
        model = arm_model(template, arm)
        if key in existing:
            have, have_config = current_models[key]
            if (
                have.get("modelName") == model["modelName"]
                and (have.get("parameters") or {}) == model["parameters"]
                and not have.get("custom")
                and have_config == template["modelConfigKey"]
            ):
                print(f"variation {key}: already exists")
                continue
            status, body = call(
                "PATCH",
                f"{base}/{CONFIG_KEY}/variations/{key}",
                {"name": arm["name"], "model": model, "modelConfigKey": template["modelConfigKey"]},
            )
            print(f"variation {key}: updated to {model['modelName']}, HTTP {status}")
            if status != 200:
                sys.exit(f"updating variation {key} failed: {message(body)}")
            continue
        status, body = call("POST", f"{base}/{CONFIG_KEY}/variations", {
            "key": key,
            "name": arm["name"],
            "messages": template["messages"],
            "model": model,
            "modelConfigKey": template["modelConfigKey"],
            "tools": template["tools"],
        })
        print(f"variation {key}: HTTP {status}")
        if status not in (200, 201):
            sys.exit(f"creating variation {key} failed: {message(body)}")

    status, created = call("GET", f"{base}/{CONFIG_KEY}", None)
    if status != 200:
        sys.exit(f"read-back failed: HTTP {status} {message(created)}")
    variations = {v["key"]: v for v in created.get("variations", [])}
    expected_tools = sorted(t["key"] for t in template["tools"])
    for key, arm in ARMS.items():
        v = variations.get(key)
        if v is None:
            sys.exit(f"read-back: variation {key} missing")
        tools = sorted(t["key"] for t in v.get("tools") or [])
        checks = {
            "model": v["model"]["modelName"] == template["model"]["modelName"],
            "parameters": v["model"]["parameters"] == arm_model(template, arm)["parameters"],
            "custom": not v["model"].get("custom"),
            "messages": v["messages"] == template["messages"],
            "tools": tools == expected_tools,
        }
        print(f"read-back {key}: {len(tools)} tools, parameters={v['model']['parameters']}, "
              + ", ".join(f"{name} {'ok' if ok else 'MISMATCH'}" for name, ok in checks.items()))
        if not all(checks.values()):
            sys.exit(1)


if __name__ == "__main__":
    main()
