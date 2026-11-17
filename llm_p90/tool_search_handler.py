"""A Claude Messages handler that can defer tool definitions behind tool search.

The Agent Management Claude handler sends every attached tool up front and has no way
to set ``defer_loading``. This handler serves both arms of the tool-search
experiment, so the arms differ only in what the variation says:

* without ``model.parameters.tool_search``, every attached tool goes out up front,
  the same request the stock handler builds;
* with ``model.parameters.tool_search`` set to ``"bm25"`` or ``"regex"``, Anthropic's
  tool search tool goes first and every attached tool is sent with
  ``defer_loading: true``, so the model searches before it calls.

The switch is a model parameter, not ``model.custom``: evaluations from code build the
handler config from the model name and parameters only and drop ``model.custom``
(launchdarkly_ai_server/evaluations/runner.py, ``_build_handler_config``). Both handlers
forward parameters through the same allowlist, which has no ``tool_search``, so the key
never reaches the provider. Prompt building and parameter forwarding
reuse the stock handler's own functions, so apart from ``defer_loading`` both arms
send exactly the request the stock handler would build from the same variation.

Tool search reference:
https://platform.claude.com/docs/en/agents-and-tools/tool-use/tool-search-tool
"""

from __future__ import annotations

import inspect
import json
import os
from collections.abc import Callable, Mapping
from contextvars import ContextVar
from typing import Any

from launchdarkly_ai_claude_messages.handler import (
    _build_messages,
    _forwarded_parameters,
)
from launchdarkly_ai_claude_messages.spans import (
    RawRunUsage,
    fail_span,
    finish_model_span,
    finish_root_span,
    parent_context_of,
    raw_usage_of,
    start_model_span,
    start_root_span,
    start_tool_span,
    succeed_span,
)
from launchdarkly_ai_server import (
    create_handler,
    end_unfinished_spans,
    to_semconv_finish_reason,
    tool_result_string,
)

from llm_p90.bfcl_loop import EPISODE_MESSAGES, STEP_LIMIT_MESSAGE

#: Same cap as the stock Claude handler, so a runaway turn force-quits identically.
MAX_STEPS = 10
#: Resumptions of a paused server-side search allowed per turn. Kept apart from MAX_STEPS
#: so a pause never eats the client tool-call budget the all-tools arm also has.
MAX_PAUSES = 10

SEARCH_TOOLS = {
    "bm25": {"type": "tool_search_tool_bm25_20251119", "name": "tool_search_tool_bm25"},
    "regex": {"type": "tool_search_tool_regex_20251119", "name": "tool_search_tool_regex"},
}

#: Opt-in trace for debugging: set LLM_P90_TRACE to a file path and every response's stop
#: reason, block types, search queries, and returned tool names are appended as JSON lines.
TRACE_PATH = os.environ.get("LLM_P90_TRACE")


def _trace(response: Any, steps: int) -> None:
    if not TRACE_PATH:
        return
    blocks = []
    for b in response.content:
        entry: dict[str, Any] = {"type": b.type}
        if b.type == "server_tool_use":
            entry["input"] = getattr(b, "input", None)
        elif b.type == "tool_search_tool_result":
            content = getattr(b, "content", None)
            refs = getattr(content, "tool_references", None)
            if refs is None and isinstance(content, dict):
                refs = content.get("tool_references")
            entry["found"] = [getattr(r, "tool_name", None) or (r.get("tool_name") if isinstance(r, dict) else None) for r in (refs or [])]
            entry["error"] = getattr(content, "error_code", None) or (content.get("error_code") if isinstance(content, dict) else None)
        elif b.type == "tool_use":
            entry["name"] = b.name
        elif b.type == "text":
            entry["text"] = b.text[:200]
        blocks.append(entry)
    usage = getattr(response, "usage", None)
    record = {"step": steps, "stop_reason": response.stop_reason, "blocks": blocks,
              "output_tokens": getattr(usage, "output_tokens", None)}
    with open(TRACE_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=str) + "\n")


#: Per-task tally of tool searches, for the audit log. The runner installs a fresh
#: ``{"searches": 0}`` before each task. It is a mutable dict because the SDK runs the
#: handler in a copied context, where a plain value set here would never reach the runner.
SEARCH_TALLY: ContextVar[dict[str, int] | None] = ContextVar("tool_search_tally", default=None)


def tool_search_mode(config: Mapping[str, Any]) -> str | None:
    parameters = (config.get("model") or {}).get("parameters") or {}
    mode = parameters.get("tool_search") if isinstance(parameters, Mapping) else None
    if mode is None:
        return None
    if mode not in SEARCH_TOOLS:
        raise ValueError(f"model.parameters.tool_search must be one of {sorted(SEARCH_TOOLS)}")
    return mode


def build_tools(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    """The ``tools`` array for one request, in the order the variation attaches them."""
    attached = config.get("tools") or {}
    mode = tool_search_mode(config)
    tools = [
        {
            "name": name,
            "description": tool.get("description", ""),
            "input_schema": tool.get("parameters", {}),
            **({"defer_loading": True} if mode else {}),
        }
        for name, tool in attached.items()
    ]
    return [dict(SEARCH_TOOLS[mode]), *tools] if mode else tools


def create_tool_search_handler(client: Any | None = None) -> Any:
    """Return a provider handler. ``client`` is an ``AsyncAnthropic``; tests pass a fake."""
    if client is None:
        import anthropic

        client = anthropic.AsyncAnthropic()

    async def run(
        config: dict[str, Any],
        user_input: str = "",
        tool_handlers: Mapping[str, Callable[..., Any]] | None = None,
        variables: dict[str, Any] | None = None,
        history: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        handlers = tool_handlers or {}
        vs = variables or {}
        tools = build_tools(config)
        # The stock handler's private helpers: pinned launchdarkly-ai-claude-messages 1.0.0.
        episode = EPISODE_MESSAGES.get()
        if episode:
            # Later turn of a BFCL episode: continue the full content-block conversation, so
            # tools discovered by search earlier stay loaded. The text ``history`` is ignored.
            _, system = _build_messages(config, user_input, vs)
            conversation = [*episode, {"role": "user", "content": user_input}]
        else:
            conversation, system = _build_messages(config, user_input, vs, history=history)
        request: dict[str, Any] = _forwarded_parameters(config)
        request["max_tokens"] = request.pop("max_tokens", 1024)
        request["model"] = config["model"]["name"]
        if system:
            request["system"] = system
        if tools:
            request["tools"] = tools

        run_usage = RawRunUsage()
        root = start_root_span(config, vs)
        parent = parent_context_of(root)
        open_span: Any = None
        steps = pauses = 0
        try:
            while True:
                open_span = start_model_span(config, parent)
                response = await client.messages.create(**request, messages=conversation)
                _trace(response, steps)
                raw_usage = raw_usage_of(getattr(response, "usage", None))
                run_usage.add_turn(raw_usage)
                finish_model_span(
                    open_span, config, raw_usage, to_semconv_finish_reason(response.stop_reason)
                )
                open_span = None
                if (tally := SEARCH_TALLY.get()) is not None:
                    tally["searches"] += sum(
                        1 for block in response.content if block.type == "server_tool_use"
                    )

                # Server tool blocks (server_tool_use, tool_search_tool_result) go back unchanged.
                conversation.append({"role": "assistant", "content": response.content})
                if response.stop_reason == "pause_turn":
                    if pauses >= MAX_PAUSES:
                        raise RuntimeError(f"{STEP_LIMIT_MESSAGE} (pauses)")
                    pauses += 1
                    continue
                if response.stop_reason != "tool_use":
                    if episode is not None:
                        episode[:] = conversation
                    output = "".join(b.text for b in response.content if b.type == "text")
                    finish_root_span(root, config, run_usage.total)
                    succeed_span(root)
                    root = None
                    return {"output": output, "usage": run_usage.total}
                if steps >= MAX_STEPS:
                    raise RuntimeError(f"{STEP_LIMIT_MESSAGE} ({MAX_STEPS})")
                steps += 1

                results = []
                for block in response.content:
                    if block.type != "tool_use":
                        continue
                    handler = handlers.get(block.name)
                    if handler is None:
                        raise ValueError(f'No handler registered for tool "{block.name}"')
                    open_span = start_tool_span(block.name, block.id, parent)
                    result = handler(block.input)
                    if inspect.isawaitable(result):
                        result = await result
                    succeed_span(open_span)
                    open_span = None
                    results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": tool_result_string(result),
                        }
                    )
                conversation.append({"role": "user", "content": results})
        except BaseException as exc:
            if open_span is not None:
                fail_span(open_span, exc)
                open_span = None
            if root is not None:
                if run_usage.reported:
                    finish_root_span(root, config, run_usage.total)
                fail_span(root, exc)
                root = None
            raise
        finally:
            end_unfinished_spans(open_span, root)

    return create_handler(("Anthropic", "messages"), run)
