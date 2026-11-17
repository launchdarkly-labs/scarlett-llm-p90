"""The tool-creation script against a fake LaunchDarkly API. No network."""

import importlib.util
import json

import pytest
from bfcl_policies import ROOT

spec = importlib.util.spec_from_file_location(
    "create_bfcl_tools", ROOT / "scripts" / "create_bfcl_tools.py"
)
create_bfcl_tools = importlib.util.module_from_spec(spec)
spec.loader.exec_module(create_bfcl_tools)

TOOLS = json.loads((ROOT / "tools" / "bfcl_tools.json").read_text())["tools"]


class FakeApi:
    def __init__(self, existing=(), fail=(), drop_tools=False):
        definitions = {tool["key"]: tool for tool in TOOLS}
        self.tools = {
            key: {
                **definitions.get(key, {}),
                "key": key,
                "version": 1,
                "tags": ["bfcl", "llm-p90"],
            }
            for key in existing
        }
        self.drop_tools = drop_tools
        self.fail = set(fail)
        self.calls = []

    def __call__(self, method, path, body):
        self.calls.append((method, path, body))
        if method == "GET":
            offset = int(path.split("offset=")[1])
            items = list(self.tools.values())[offset : offset + 100]
            return 200, {"items": items, "totalCount": len(self.tools)}
        if method == "POST":
            if body["key"] in self.fail:
                return 400, {"message": "bad schema"}
            self.tools[body["key"]] = {**body, "version": 1}
            return 201, body
        if method == "PATCH":
            if self.drop_tools:
                return 200, {"key": "variation"}
            return 200, {"tools": body["tools"]}
        raise AssertionError(method)


def test_creates_every_tool_with_its_exact_definition_and_tags():
    api = FakeApi()

    report = create_bfcl_tools.create_tools(api, "llm-p90-experiment", TOOLS)

    assert len(report["created"]) == 128
    posted = {body["key"]: body for method, _, body in api.calls if method == "POST"}
    first = TOOLS[0]
    assert posted[first["key"]]["schema"] == first["schema"]
    assert posted[first["key"]]["description"] == first["description"]
    assert posted[first["key"]]["tags"] == ["bfcl", "llm-p90"]


def test_rerun_skips_tools_that_already_exist_across_pages():
    api = FakeApi(existing=[tool["key"] for tool in TOOLS])

    report = create_bfcl_tools.create_tools(api, "llm-p90-experiment", TOOLS)

    assert report["created"] == []
    assert len(report["skipped"]) == 128
    assert sum(1 for method, *_ in api.calls if method == "GET") == 2


def test_rerun_refuses_a_stale_existing_definition():
    api = FakeApi(existing=[TOOLS[0]["key"]])
    api.tools[TOOLS[0]["key"]]["description"] = "stale"

    report = create_bfcl_tools.create_tools(api, "llm-p90-experiment", TOOLS[:1])

    assert report["created"] == []
    assert report["skipped"] == []
    assert report["failed"][0]["status"] == "definition_mismatch"


def test_failures_are_reported_not_hidden():
    api = FakeApi(fail=[TOOLS[3]["key"]])

    report = create_bfcl_tools.create_tools(api, "llm-p90-experiment", TOOLS)

    assert report["failed"] == [{"key": TOOLS[3]["key"], "status": 400, "error": {"message": "bad schema"}}]
    assert len(report["created"]) == 127


def test_attach_sends_every_key_in_one_patch_per_variation():
    api = FakeApi(existing=[tool["key"] for tool in TOOLS])
    keys = [tool["key"] for tool in TOOLS]

    attached = create_bfcl_tools.attach_tools(
        api, "llm-p90-experiment", "bfcl-model-comparison", ["claude-haiku-5-5", "claude-sonnet-5-5"], keys
    )

    patches = [(path, body) for method, path, body in api.calls if method == "PATCH"]
    assert attached == ["claude-haiku-5-5", "claude-sonnet-5-5"]
    assert patches[0][0].endswith("/ai-configs/bfcl-model-comparison/variations/claude-haiku-5-5")
    assert patches[0][1]["tools"] == [{"key": key, "version": 1} for key in keys]


def test_attach_fails_when_the_api_accepts_the_patch_but_attaches_nothing():
    api = FakeApi(existing=[tool["key"] for tool in TOOLS], drop_tools=True)
    keys = [tool["key"] for tool in TOOLS]

    with pytest.raises(RuntimeError, match="reports 0 of 128 tools"):
        create_bfcl_tools.attach_tools(api, "p", "bfcl-model-comparison", ["claude-sonnet-5-5"], keys)


def test_attach_refuses_keys_missing_from_the_library():
    api = FakeApi(existing=["cat"])

    with pytest.raises(RuntimeError, match="missing from the library"):
        create_bfcl_tools.attach_tools(api, "p", "c", ["v"], ["cat", "cd"])


def test_attach_stops_on_an_api_error():
    api = FakeApi(existing=["cat"])

    def broken(method, path, body):
        if method == "GET":
            return api(method, path, body)
        return 404, {"message": "variation not found"}

    with pytest.raises(RuntimeError, match="variation not found"):
        create_bfcl_tools.attach_tools(broken, "p", "c", ["missing"], ["cat"])


def test_rate_limited_requests_are_retried_with_backoff():
    api = FakeApi()
    responses = iter([(429, {"code": "rate_limited"}), (429, {"code": "rate_limited"})])
    waits = []

    def flaky(method, path, body):
        if method == "POST":
            nxt = next(responses, None)
            if nxt:
                return nxt
        return api(method, path, body)

    http = create_bfcl_tools.with_rate_limit_retries(flaky, sleep=waits.append)
    report = create_bfcl_tools.create_tools(http, "p", TOOLS[:3])

    assert len(report["created"]) == 3
    assert report["failed"] == []
    assert waits == [1, 2]


def test_rate_limit_retries_give_up_and_report():
    def always_limited(method, path, body):
        if method == "GET":
            return 200, {"items": [], "totalCount": 0}
        return 429, {"code": "rate_limited"}

    waits = []
    http = create_bfcl_tools.with_rate_limit_retries(always_limited, attempts=3, sleep=waits.append)
    report = create_bfcl_tools.create_tools(http, "p", TOOLS[:1])

    assert report["failed"][0]["status"] == 429
    assert waits == [1, 2]


def test_non_json_error_bodies_report_instead_of_raising():
    assert create_bfcl_tools.decode_body(b"<html>502 Bad Gateway</html>") == {
        "message": "<html>502 Bad Gateway</html>"
    }
    assert create_bfcl_tools.decode_body(b'{"message": "x"}') == {"message": "x"}
    assert create_bfcl_tools.decode_body(b"") is None


def load_setup_script(monkeypatch, responses):
    """Import create_tool_search_config with a scripted HTTP layer; return (module, calls)."""
    import sys

    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    monkeypatch.setenv("LD_OPERATOR_TOKEN", "test-token")
    spec = importlib.util.spec_from_file_location(
        "create_tool_search_config", ROOT / "scripts" / "create_tool_search_config.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    calls = []

    def http(method, path, body):
        calls.append((method, path))
        return responses.pop(0)

    monkeypatch.setattr(module, "http_with_token", lambda token, beta=False: http)
    monkeypatch.setattr(module, "load_dotenv", lambda: None)
    sys.modules.pop("create_tool_search_config", None)
    return module, calls


def test_setup_names_the_missing_source_variation(monkeypatch):
    source = {"variations": [{"key": "claude-sonnet-5-5"}]}
    module, _ = load_setup_script(monkeypatch, [(200, source)])
    with pytest.raises(SystemExit, match="claude-haiku-5-5 not found.*claude-sonnet-5-5"):
        module.main()


def test_setup_skips_an_existing_config_without_reposting(monkeypatch):
    template = {
        "key": "claude-haiku-5-5",
        "messages": [{"role": "system", "content": "m"}],
        "model": {"modelName": "claude-haiku-5-5", "parameters": {"max_tokens": 2048}},
        "modelConfigKey": "Anthropic.claude-haiku-5-5",
        "tools": [{"key": "cd", "version": 1}],
    }

    def arm(key, extra):
        parameters = {**template["model"]["parameters"], **extra}
        return {**template, "key": key, "model": {**template["model"], "parameters": parameters}}

    created = {"variations": [arm("all-tools", {}), arm("tool-search", {"tool_search": "bm25"})]}
    module, calls = load_setup_script(
        monkeypatch, [(200, {"variations": [template]}), (200, created), (200, created)]
    )
    module.main()
    assert [method for method, _ in calls] == ["GET", "GET", "GET"]


def test_setup_moves_a_stale_switch_into_parameters(monkeypatch):
    template = {
        "key": "claude-haiku-5-5",
        "messages": [{"role": "system", "content": "m"}],
        "model": {"modelName": "claude-haiku-5-5", "parameters": {"max_tokens": 2048}},
        "modelConfigKey": "Anthropic.claude-haiku-5-5",
        "tools": [{"key": "cd", "version": 1}],
    }
    stale = {"variations": [
        {**template, "key": "all-tools"},
        {**template, "key": "tool-search",
         "model": {**template["model"], "custom": {"tool_search": "bm25"}}},
    ]}
    fixed_model = {**template["model"], "parameters": {"max_tokens": 2048, "tool_search": "bm25"}}
    fixed = {"variations": [
        {**template, "key": "all-tools"},
        {**template, "key": "tool-search", "model": fixed_model},
    ]}
    module, calls = load_setup_script(
        monkeypatch, [(200, {"variations": [template]}), (200, stale), (200, {}), (200, fixed)]
    )
    module.main()
    assert ("PATCH", "/projects/llm-p90-experiment/ai-configs/bfcl-tool-search/variations/tool-search") in calls
