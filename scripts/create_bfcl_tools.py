"""Create the 128 BFCL tools in the Agent Management tool library, then attach them.

Reads tools/bfcl_tools.json and creates each tool with LaunchDarkly's
Create AI tool REST endpoint, using LD_OPERATOR_TOKEN and LD_PROJECT_KEY from
.env. Tools that already exist are verified before they are skipped, so the
script cannot silently attach a stale definition.
With --attach, it also attaches every tool to the named variations of the
config in one request each.

    uv run python scripts/create_bfcl_tools.py
    uv run python scripts/create_bfcl_tools.py --attach claude-haiku-5-5 claude-sonnet-5-5
"""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

API_ROOT = "https://app.launchdarkly.com/api/v2"
TOOLS_FILE = Path(__file__).parents[1] / "tools" / "bfcl_tools.json"
TAGS = ["bfcl", "llm-p90"]

Http = Callable[[str, str, dict[str, Any] | None], tuple[int, Any]]


def decode_body(raw: bytes) -> Any:
    """JSON when the body is JSON; otherwise the text as ``{"message": ...}`` so a 502
    page or a plain-text 429 reports its status instead of raising."""
    if not raw:
        return None
    try:
        return json.loads(raw)
    except ValueError:
        return {"message": raw.decode(errors="replace")[:500]}


def http_with_token(token: str, *, beta: bool = False) -> Http:
    """``beta`` sends ``LD-API-Version: beta``, which the Agent Management config endpoints need."""
    headers = {"Authorization": token, "Content-Type": "application/json"}
    if beta:
        headers["LD-API-Version"] = "beta"

    def send(method: str, path: str, body: dict[str, Any] | None) -> tuple[int, Any]:
        request = urllib.request.Request(
            f"{API_ROOT}{path}",
            method=method,
            data=None if body is None else json.dumps(body).encode(),
            headers=headers,
        )
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return response.status, decode_body(response.read())
        except urllib.error.HTTPError as error:
            return error.code, decode_body(error.read())

    return send


def with_rate_limit_retries(
    http: Http, *, attempts: int = 6, sleep: Callable[[float], None] = time.sleep
) -> Http:
    """Retry HTTP 429 responses with exponential backoff: 1, 2, 4, 8, 16 seconds."""

    def send(method: str, path: str, body: dict[str, Any] | None) -> tuple[int, Any]:
        for attempt in range(attempts):
            status, response = http(method, path, body)
            if status != 429 or attempt == attempts - 1:
                return status, response
            sleep(2**attempt)
        return status, response

    return send


def existing_tools(http: Http, project: str) -> dict[str, dict[str, Any]]:
    """Map of tool key to its latest definition, across every page of the library."""
    tools: dict[str, dict[str, Any]] = {}
    offset = 0
    while True:
        status, page = http("GET", f"/projects/{project}/ai-tools?limit=100&offset={offset}", None)
        if status != 200:
            raise RuntimeError(f"listing tools failed: HTTP {status} {page}")
        items = page.get("items", [])
        tools.update({item["key"]: item for item in items})
        offset += len(items)
        if not items or offset >= page.get("totalCount", 0):
            return tools


def create_tools(http: Http, project: str, tools: list[dict[str, Any]]) -> dict[str, list]:
    existing = existing_tools(http, project)
    report: dict[str, list] = {"created": [], "skipped": [], "failed": []}
    for tool in tools:
        if tool["key"] in existing:
            current = existing[tool["key"]]
            expected = {
                "description": tool["description"],
                "schema": tool["schema"],
                "tags": TAGS,
            }
            actual = {field: current.get(field) for field in expected}
            if actual != expected:
                report["failed"].append(
                    {
                        "key": tool["key"],
                        "status": "definition_mismatch",
                        "error": {"expected": expected, "actual": actual},
                    }
                )
            else:
                report["skipped"].append(tool["key"])
            continue
        body = {
            "key": tool["key"],
            "description": tool["description"],
            "schema": tool["schema"],
            "tags": TAGS,
        }
        status, response = http("POST", f"/projects/{project}/ai-tools", body)
        if status == 201:
            report["created"].append(tool["key"])
        elif status == 409:
            report["skipped"].append(tool["key"])
        else:
            report["failed"].append({"key": tool["key"], "status": status, "error": response})
    return report


def attach_tools(
    http: Http, project: str, config: str, variations: list[str], keys: list[str]
) -> list[str]:
    """Attach every key, at its latest version, then confirm the variation lists them all."""
    existing = existing_tools(http, project)
    missing = [key for key in keys if key not in existing]
    if missing:
        raise RuntimeError(f"tools missing from the library: {missing}")
    tools = [{"key": key, "version": existing[key].get("version", 1)} for key in keys]
    attached = []
    for variation in variations:
        status, response = http(
            "PATCH",
            f"/projects/{project}/ai-configs/{config}/variations/{variation}",
            {"tools": tools, "comment": f"Attach all {len(keys)} BFCL tools"},
        )
        if status != 200:
            raise RuntimeError(f"attaching tools to {variation} failed: HTTP {status} {response}")
        # A 200 alone is not proof: check the variation now carries every tool.
        got = {tool.get("key") for tool in (response or {}).get("tools") or []}
        if not set(keys) <= got:
            raise RuntimeError(
                f"{variation} reports {len(got)} of {len(keys)} tools after the update"
            )
        attached.append(variation)
    return attached


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="bfcl-model-comparison")
    parser.add_argument("--attach", nargs="*", default=[], metavar="VARIATION")
    args = parser.parse_args()

    token = os.getenv("LD_OPERATOR_TOKEN")
    project = os.getenv("LD_PROJECT_KEY")
    if not token or not project:
        parser.error("LD_OPERATOR_TOKEN and LD_PROJECT_KEY are required")

    tools = json.loads(TOOLS_FILE.read_text())["tools"]
    http = with_rate_limit_retries(http_with_token(token))
    report = create_tools(http, project, tools)
    print(
        f"created {len(report['created'])}, already present {len(report['skipped'])}, "
        f"failed {len(report['failed'])} of {len(tools)} tools"
    )
    for failure in report["failed"]:
        print(f"  failed {failure['key']}: HTTP {failure['status']} {failure['error']}")
    if report["failed"]:
        raise SystemExit(1)
    if args.attach:
        keys = [tool["key"] for tool in tools]
        for variation in attach_tools(http, project, args.config, args.attach, keys):
            print(f"attached {len(keys)} tools to {args.config}/{variation}")


if __name__ == "__main__":
    main()
