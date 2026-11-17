"""BFCL multi-turn episodes over native function calling.

BFCL multi-turn tasks are interactive. For each user turn the model calls
functions, BFCL executes them against its simulated backends, and the results
go back to the model until it stops calling functions. Only then does the next
user turn arrive. The checker replays the calls and compares state and
responses against the ground truth after every turn.

Everything the model sees comes from the Agent Management config: the model, its
parameters, and the tool definitions attached to the variation. The provider
handler sends those tools through the provider's native tool parameter and
runs the call loop. This module supplies only the tool implementations, which
execute each call on BFCL's backend for the task in progress, and the turn
loop around them. BFCL-defined pieces (call formatting, the executor, the
checker) come from the pinned ``bfcl-eval`` package.
"""

from __future__ import annotations

import copy
import json
import re
from collections.abc import Awaitable, Callable, Iterable, Mapping
from contextvars import ContextVar
from dataclasses import dataclass, field
from functools import cache

from bfcl_eval.constants.eval_config import MULTI_TURN_FUNC_DOC_PATH
from bfcl_eval.constants.executable_backend_config import (
    MULTI_TURN_FUNC_DOC_FILE_MAPPING,
)
from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_utils import (
    execute_multi_turn_func_call,
)
from bfcl_eval.model_handler.utils import convert_to_function_call
from bfcl_eval.utils import populate_test_cases_with_predefined_functions

# Text the Claude handler raises when the model keeps calling tools past its
# per-invocation step limit. BFCL also ends such a task as a failure.
STEP_LIMIT_MESSAGE = "Tool loop exceeded the maximum number of steps"

#: The full Messages API conversation of the current attempt, for handlers that keep
#: content blocks across turns. ``run_episode`` installs a fresh list per attempt. The
#: stock Claude handler ignores it and rebuilds history from the text ``history``; the
#: tool-search handler uses it, so tools found by search in one turn stay loaded in the
#: next, as they do in an app that passes back ``response.content``. It is a mutable list
#: because the SDK runs handlers in a copied context.
EPISODE_MESSAGES: ContextVar[list[dict[str, object]] | None] = ContextVar(
    "episode_messages", default=None
)

TurnCall = Callable[[str, list[dict[str, str]]], Awaitable[str]]


def tool_key(function_name: str) -> str:
    """Agent Management tool key for a BFCL function name.

    The AI SDK's evaluations require lowercase tool keys, and ten BFCL
    vehicle functions are camelCase (``lockDoors``). Those become snake_case
    (``lock_doors``); every other name is unchanged.
    """
    return re.sub(r"(?<!^)(?=[A-Z])", "_", function_name).lower()


@cache
def _function_names() -> dict[str, str]:
    names: dict[str, str] = {}
    for doc_file in MULTI_TURN_FUNC_DOC_FILE_MAPPING.values():
        path = MULTI_TURN_FUNC_DOC_PATH / doc_file
        for line in path.read_text().splitlines():
            if line.strip():
                name = json.loads(line)["name"]
                names[tool_key(name)] = name
    return names


def function_name(key: str) -> str:
    """The BFCL function a tool key runs. Unknown keys pass through unchanged."""
    return _function_names().get(key, key)


def function_docs(test_entry: Mapping[str, object]) -> list[dict[str, object]]:
    """Function docs for a multi-turn entry, populated exactly as BFCL does."""
    entry = copy.deepcopy(dict(test_entry))
    populate_test_cases_with_predefined_functions([entry])
    docs = entry.get("function")
    if not isinstance(docs, list) or not docs:
        raise ValueError(f"no BFCL function docs for {test_entry.get('id')}")
    return docs


@dataclass
class BackendSession:
    """BFCL's simulated backends for one attempt at one task.

    BFCL keeps backend instances in module globals keyed by model name and
    task id, so ``scope`` must be unique per attempt for a repair attempt to
    start from the task's initial state.
    """

    test_entry: Mapping[str, object]
    scope: str
    decoded_turns: list[list[list[str]]] = field(default_factory=list)
    transcript: list[str] = field(default_factory=list)

    def execute(self, name: str, arguments: Mapping[str, object]) -> str:
        call = convert_to_function_call({name: dict(arguments)})[0]
        entry_id = str(self.test_entry["id"])
        category = entry_id.rsplit("_", 1)[0]
        results, _ = execute_multi_turn_func_call(
            [call],
            self.test_entry.get("initial_config", {}),
            self.test_entry["involved_classes"],
            self.scope,
            entry_id,
            long_context="long_context" in category or "composite" in category,
            is_evaL_run=False,
        )
        result = results[0]
        # Each call is its own step. BFCL's checker merges the steps of a turn,
        # so grouping does not change the verdict.
        self.decoded_turns[-1].append([call])
        self.transcript.append(f"{call} -> {result}")
        return result


_SESSION: ContextVar[BackendSession | None] = ContextVar("bfcl_session", default=None)


def tool_implementations(tool_keys: Iterable[str]) -> dict[str, Callable[[dict], str]]:
    """One implementation per tool attached to the config.

    Each executes the call on the backend of the episode currently running in
    this context. A call that fails returns BFCL's error text to the model
    instead of raising, as BFCL's own harness does.
    """

    def implementation_for(key: str) -> Callable[[dict], str]:
        name = function_name(key)

        def run(arguments: dict) -> str:
            session = _SESSION.get()
            if session is None:
                raise RuntimeError("tool called outside a BFCL episode")
            return session.execute(name, arguments or {})

        run.__name__ = key
        return run

    return {key: implementation_for(key) for key in tool_keys}


@dataclass
class Episode:
    """One pass through every turn of a BFCL task."""

    decoded_turns: list[list[list[str]]]
    force_quit: bool = False

    def as_output(self) -> str:
        """The generated-output JSON that ``llm_p90.scoring`` hands to BFCL's checker."""
        return json.dumps({"turns": self.decoded_turns}, separators=(",", ":"))


def _turn_text(turn_messages: object) -> str:
    if not isinstance(turn_messages, list):
        raise TypeError("BFCL turn must be an array of messages")
    return "\n".join(
        str(message.get("content", ""))
        for message in turn_messages
        if isinstance(message, dict) and message.get("role", "user") == "user"
    )


async def run_episode(
    test_entry: Mapping[str, object],
    call_turn: TurnCall,
    *,
    scope: str,
    repair_note: str | None = None,
) -> Episode:
    """Play every turn of ``test_entry``; tool calls execute on BFCL's backends.

    ``call_turn(user_text, history)`` runs one user turn through the provider
    handler, which loops over native tool calls until the model stops, and
    returns the model's final text. The SDK history carries text, so each
    earlier turn's calls and results are carried forward as text in the
    assistant message.

    ``repair_note`` is prepended to the first user turn of a repair attempt.
    It must carry only official checker feedback, never the expected answer.
    """
    questions = test_entry["question"]
    if not isinstance(questions, list) or not questions:
        raise ValueError(f"{test_entry.get('id')} has no question turns")

    session = BackendSession(test_entry=test_entry, scope=scope)
    token = _SESSION.set(session)
    messages_token = EPISODE_MESSAGES.set([])
    history: list[dict[str, str]] = []
    force_quit = False
    try:
        for turn_index, turn_messages in enumerate(questions):
            user_text = _turn_text(turn_messages)
            if turn_index == 0 and repair_note:
                user_text = f"{repair_note}\n\n{user_text}"
            session.decoded_turns.append([])
            session.transcript = []
            try:
                final_text = await call_turn(user_text, list(history))
            except RuntimeError as error:
                if STEP_LIMIT_MESSAGE not in str(error):
                    raise
                force_quit = True
                break
            calls = "\n".join(session.transcript)
            assistant = f"{calls}\n\n{final_text}".strip() if calls else final_text
            history.append({"role": "user", "content": user_text})
            history.append({"role": "assistant", "content": assistant or "(no reply)"})
    finally:
        EPISODE_MESSAGES.reset(messages_token)
        _SESSION.reset(token)
    return Episode(decoded_turns=session.decoded_turns, force_quit=force_quit)


def repair_note_from(validation: Mapping[str, object]) -> str:
    """Build the repair message from official checker feedback only.

    The checker's ``details`` can hold ground-truth execution results, so only
    the error type and message are used.
    """
    message = validation.get("error_message") or validation.get("error") or "invalid"
    if isinstance(message, list):
        message = "; ".join(str(item) for item in message)
    error_type = validation.get("error_type")
    prefix = f"[{error_type}] " if error_type else ""
    return (
        "The official BFCL evaluator rejected your previous attempt at this "
        f"task: {prefix}{message}. Start the task again from the beginning."
    )
