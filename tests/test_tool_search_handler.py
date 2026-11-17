"""The tool-search handler: request shape per arm and the tool loop. No network, no spend."""

import asyncio
import json
from types import SimpleNamespace

import pytest

from llm_p90.bfcl_loop import STEP_LIMIT_MESSAGE
from llm_p90.tool_search_handler import (
    SEARCH_TALLY,
    build_tools,
    create_tool_search_handler,
)

TOOLS = {
    "cd": {"description": "Change directory", "parameters": {"type": "object"}},
    "ls": {"description": "List files", "parameters": {"type": "object"}},
}


def variation(tool_search=None):
    model = {"name": "claude-sonnet-5-5", "parameters": {"max_tokens": 2048}}
    if tool_search:
        model["parameters"]["tool_search"] = tool_search
    return {
        "model": model,
        "messages": [{"role": "system", "content": "Use the available tools."}],
        "tools": TOOLS,
    }


def block(kind, **fields):
    return SimpleNamespace(type=kind, **fields)


def reply(stop_reason, *content):
    return SimpleNamespace(
        stop_reason=stop_reason,
        content=list(content),
        usage=SimpleNamespace(input_tokens=100, output_tokens=10),
    )


class FakeClient:
    def __init__(self, replies):
        self.replies = list(replies)
        self.requests = []
        self.messages = SimpleNamespace(create=self.create)

    async def create(self, **request):
        self.requests.append({**request, "messages": list(request["messages"])})
        return self.replies.pop(0)


def run(handler, config, handlers=None, history=None):
    return asyncio.run(handler(config, "list the files", handlers or {}, {}, history))


def test_all_tools_arm_sends_every_tool_up_front():
    tools = build_tools(variation())
    assert [t["name"] for t in tools] == ["cd", "ls"]
    assert not any("defer_loading" in t for t in tools)


def test_tool_search_arm_defers_every_attached_tool_behind_the_search_tool():
    tools = build_tools(variation("bm25"))
    assert tools[0] == {"type": "tool_search_tool_bm25_20251119", "name": "tool_search_tool_bm25"}
    assert "defer_loading" not in tools[0]
    assert [t["name"] for t in tools[1:]] == ["cd", "ls"]
    assert all(t["defer_loading"] is True for t in tools[1:])


def test_unknown_tool_search_mode_is_rejected():
    with pytest.raises(ValueError):
        build_tools(variation("embeddings"))


def test_request_takes_model_max_tokens_and_system_from_the_variation():
    client = FakeClient([reply("end_turn", block("text", text="done"))])
    result = run(create_tool_search_handler(client), variation())
    request = client.requests[0]
    assert request["model"] == "claude-sonnet-5-5"
    assert request["max_tokens"] == 2048
    assert request["system"] == "Use the available tools."
    assert "temperature" not in request
    assert result == {"output": "done", "usage": {"input_tokens": 100, "output_tokens": 10}}


def test_search_then_call_runs_the_tool_and_returns_server_blocks_unchanged():
    search = block("server_tool_use", id="srvtoolu_1", name="tool_search_tool_bm25", input={})
    found = block("tool_search_tool_result", tool_use_id="srvtoolu_1", content={})
    call = block("tool_use", id="toolu_1", name="ls", input={"a": False})
    client = FakeClient(
        [reply("tool_use", search, found, call), reply("end_turn", block("text", text="ok"))]
    )
    calls = []

    def ls(arguments):
        calls.append(arguments)
        return "file.txt"

    tally = {"searches": 0}
    token = SEARCH_TALLY.set(tally)
    try:
        result = run(create_tool_search_handler(client), variation("bm25"), {"ls": ls})
    finally:
        SEARCH_TALLY.reset(token)
    searches = tally["searches"]

    assert calls == [{"a": False}]
    assert searches == 1
    second = client.requests[1]["messages"]
    assert second[1]["content"] == [search, found, call]
    assert second[2] == {
        "role": "user",
        "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": "file.txt"}],
    }
    assert result["usage"] == {"input_tokens": 200, "output_tokens": 20}


def test_history_comes_before_the_new_user_turn():
    client = FakeClient([reply("end_turn", block("text", text="ok"))])
    history = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
    run(create_tool_search_handler(client), variation(), history=history)
    assert client.requests[0]["messages"] == [
        *history,
        {"role": "user", "content": "list the files"},
    ]


def test_pause_turn_continues_without_tool_results():
    paused = block("server_tool_use", id="srvtoolu_1", name="tool_search_tool_bm25", input={})
    client = FakeClient([reply("pause_turn", paused), reply("end_turn", block("text", text="ok"))])
    run(create_tool_search_handler(client), variation("bm25"))
    assert client.requests[1]["messages"][-1] == {"role": "assistant", "content": [paused]}


def test_step_limit_raises_the_message_the_episode_loop_force_quits_on():
    call = block("tool_use", id="toolu_1", name="ls", input={})
    client = FakeClient([reply("tool_use", call) for _ in range(12)])
    with pytest.raises(RuntimeError, match=STEP_LIMIT_MESSAGE):
        run(create_tool_search_handler(client), variation(), {"ls": lambda _: "x"})


class GroundTruthClient:
    """Plays each BFCL turn: one tool search, then that turn's ground-truth calls."""

    def __init__(self, row):
        from bfcl_policies import ground_truth_arguments

        from llm_p90.bfcl_loop import function_docs, tool_key

        docs = {d["name"]: d for d in function_docs(row["metadata"]["test_entry"])}
        self.turns = []
        for calls in json.loads(row["expected_output"]):
            parsed = [ground_truth_arguments(call, docs) for call in calls]
            self.turns.append([(tool_key(name), arguments) for name, arguments in parsed])
        self.turns.reverse()
        self.messages = SimpleNamespace(create=self.create)
        self.ids = 0

    async def create(self, **request):
        assert all(t.get("defer_loading") for t in request["tools"][1:])
        last = request["messages"][-1]
        if isinstance(last["content"], str):
            calls = self.turns.pop()
            if calls:
                search = block("server_tool_use", id="srvtoolu_x", name="tool_search_tool_bm25", input={})
                found = block("tool_search_tool_result", tool_use_id="srvtoolu_x", content={})
                uses = []
                for key, arguments in calls:
                    self.ids += 1
                    uses.append(block("tool_use", id=f"toolu_{self.ids}", name=key, input=arguments))
                return reply("tool_use", search, found, *uses)
        return reply("end_turn", block("text", text="Done."))


def test_scripted_model_resolves_every_pilot_task_through_the_tool_search_handler():
    from bfcl_policies import TOOL_KEYS, rows

    from llm_p90.bfcl_loop import run_episode, tool_implementations
    from llm_p90.scoring import score_with_bfcl

    config = variation("bm25")
    config["tools"] = {key: {"description": "", "parameters": {}} for key in TOOL_KEYS}
    implementations = tool_implementations(TOOL_KEYS)
    for row in rows("bfcl_pilot.jsonl"):
        handler = create_tool_search_handler(GroundTruthClient(row))

        async def call_turn(user_text, history, handler=handler):
            result = await handler(config, user_text, implementations, {}, history)
            return result["output"]

        scope = f"ts-{row['metadata']['source_id']}"
        episode = asyncio.run(run_episode(row["metadata"]["test_entry"], call_turn, scope=scope))
        assert score_with_bfcl(row, episode.as_output(), model_name=scope) == 1.0, row["metadata"]["source_id"]


def test_model_parameters_are_forwarded_like_the_stock_handler():
    config = variation("bm25")
    config["model"]["parameters"].update({"effort": "medium", "not_a_param": 1})
    client = FakeClient([reply("end_turn", block("text", text="ok"))])
    run(create_tool_search_handler(client), config)
    request = client.requests[0]
    assert request["output_config"] == {"effort": "medium"}
    assert request["max_tokens"] == 2048
    assert "not_a_param" not in request


def test_non_system_config_messages_reach_the_model():
    config = variation()
    config["messages"].append({"role": "user", "content": "Example request"})
    client = FakeClient([reply("end_turn", block("text", text="ok"))])
    run(create_tool_search_handler(client), config)
    request = client.requests[0]
    assert request["system"] == "Use the available tools."
    first = request["messages"][0]
    assert first["role"] == "user"
    assert "Example request" in json.dumps(first["content"])
    assert "Use the available tools." not in json.dumps(request["messages"])


def test_later_turns_keep_the_full_content_of_earlier_turns():
    """Tools found by search in turn 1 must stay loaded in turn 2, as in a real app."""
    from llm_p90.bfcl_loop import EPISODE_MESSAGES

    search = block("server_tool_use", id="srvtoolu_1", name="tool_search_tool_bm25", input={})
    found = block("tool_search_tool_result", tool_use_id="srvtoolu_1", content={})
    call = block("tool_use", id="toolu_1", name="ls", input={})
    client = FakeClient([
        reply("tool_use", search, found, call),
        reply("end_turn", block("text", text="turn one done")),
        reply("end_turn", block("text", text="turn two done")),
    ])
    handler = create_tool_search_handler(client)

    async def two_turns():
        token = EPISODE_MESSAGES.set([])
        try:
            await handler(variation("bm25"), "turn one", {"ls": lambda _: "file.txt"}, {}, [])
            text_history = [
                {"role": "user", "content": "turn one"},
                {"role": "assistant", "content": "turn one done"},
            ]
            await handler(variation("bm25"), "turn two", {"ls": lambda _: "x"}, {}, text_history)
        finally:
            EPISODE_MESSAGES.reset(token)

    asyncio.run(two_turns())
    turn_two = client.requests[2]["messages"]
    assert turn_two[0]["role"] == "user"
    assert "turn one" in json.dumps(turn_two[0]["content"])
    assert turn_two[1]["content"] == [search, found, call]
    assert turn_two[2]["content"][0]["tool_use_id"] == "toolu_1"
    assert turn_two[-1] == {"role": "user", "content": "turn two"}


def test_pauses_do_not_use_the_tool_call_budget():
    paused = block("server_tool_use", id="srvtoolu_1", name="tool_search_tool_bm25", input={})
    call = block("tool_use", id="toolu_1", name="ls", input={})
    replies = [reply("pause_turn", paused) for _ in range(5)]
    replies += [reply("tool_use", call) for _ in range(10)]
    replies.append(reply("end_turn", block("text", text="ok")))
    client = FakeClient(replies)
    result = run(create_tool_search_handler(client), variation("bm25"), {"ls": lambda _: "x"})
    assert result["output"] == "ok"


def test_tool_results_use_the_stock_formatting():
    call = block("tool_use", id="toolu_1", name="ls", input={})
    client = FakeClient([reply("tool_use", call), reply("end_turn", block("text", text="ok"))])
    run(create_tool_search_handler(client), variation(), {"ls": lambda _: {"files": ["a"]}})
    sent = client.requests[1]["messages"][-1]["content"][0]["content"]
    assert sent == '{"files": ["a"]}'


def test_spans_match_the_stock_handler_shape():
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("test")

    call = block("tool_use", id="toolu_1", name="ls", input={})
    client = FakeClient([reply("tool_use", call), reply("end_turn", block("text", text="ok"))])
    original = trace.get_tracer
    trace.get_tracer = lambda *args, **kwargs: tracer
    try:
        run(create_tool_search_handler(client), variation(), {"ls": lambda _: "x"})
    finally:
        trace.get_tracer = original

    names = sorted(span.name for span in exporter.get_finished_spans())
    assert names.count("invoke_agent") == 1
    assert sum(name.startswith("chat") for name in names) == 2
    assert sum(name.startswith("execute_tool") for name in names) == 1


def test_switch_survives_the_evaluations_config_and_never_reaches_the_provider():
    """Evaluations from code keep only model name and parameters (they drop model.custom),
    so the switch must be a parameter, and the forwarding allowlist must drop it."""
    from launchdarkly_ai_server.evaluations.runner import EvaluationsRunner

    generation = {
        "provider": "Anthropic",
        "model": "claude-sonnet-5-5",
        "parameters": {"max_tokens": 2048, "tool_search": "bm25"},
        "messages": [{"role": "system", "content": "Use the available tools."}],
    }
    config = EvaluationsRunner._build_handler_config(None, generation, [])
    assert build_tools(config)[0]["type"] == "tool_search_tool_bm25_20251119"

    client = FakeClient([reply("end_turn", block("text", text="ok"))])
    run(create_tool_search_handler(client), {**config, "tools": TOOLS})
    assert "tool_search" not in client.requests[0]
    assert all(t.get("defer_loading") for t in client.requests[0]["tools"][1:])
