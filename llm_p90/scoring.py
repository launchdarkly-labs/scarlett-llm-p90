"""Adapter from Agent Management evaluation rows to BFCL's official checker."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping

Evaluator = Callable[[object, object], Mapping[str, object]]


def _field(row: object, name: str, default: object = None) -> object:
    if isinstance(row, Mapping):
        return row.get(name, default)
    return getattr(row, name, default)


def _decode_json(value: object) -> object:
    if isinstance(value, str):
        return json.loads(value)
    return value


def _calls_for_bfcl(output: str) -> list[dict[str, object]]:
    payload = json.loads(output)
    calls = payload.get("calls") if isinstance(payload, dict) else payload
    if not isinstance(calls, list):
        raise TypeError("output must contain a calls array")

    decoded: list[dict[str, object]] = []
    for call in calls:
        if not isinstance(call, dict):
            raise TypeError("each call must be an object")
        name = call.get("name")
        arguments = call.get("arguments")
        if not isinstance(name, str) or not isinstance(arguments, dict):
            raise TypeError("each call requires a name and arguments object")
        decoded.append({name: arguments})
    return decoded


def _generated_output(output: str) -> object:
    payload = json.loads(output)
    if isinstance(payload, dict) and "turns" in payload:
        turns = payload["turns"]
        if not isinstance(turns, list):
            raise TypeError("turns must be an array")
        return turns
    return _calls_for_bfcl(output)


def _official_result(
    expected: object,
    generated: object,
    *,
    metadata: Mapping[str, object],
    model_name: str,
) -> Mapping[str, object]:
    category = str(metadata.get("test_category", "simple"))
    if category.startswith("multi_turn"):
        from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_checker import (
            multi_turn_checker,
        )

        test_entry = metadata.get("test_entry")
        if not isinstance(test_entry, dict):
            raise TypeError("multi-turn rows require the original BFCL test entry")
        if not isinstance(expected, list) or not isinstance(generated, list):
            raise TypeError("BFCL expected and generated turns must be arrays")
        if len(generated) != len(expected):
            # Same verdict BFCL's eval runner gives before it calls the checker.
            return {
                "valid": False,
                "error_type": "multi_turn:force_terminated",
                "error_message": (
                    f"Model was force-terminated. It produced {len(generated)} "
                    f"of {len(expected)} turns."
                ),
            }
        return multi_turn_checker(
            generated,
            expected,
            test_entry,
            category,
            model_name,
        )

    from bfcl_eval.constants.enums import Language
    from bfcl_eval.eval_checker.ast_eval.ast_checker import ast_checker

    language_name = str(metadata.get("language", "python")).upper()
    language = Language[language_name]
    functions = metadata.get("functions")
    if not isinstance(functions, list):
        raise TypeError("dataset row metadata requires BFCL function descriptions")
    if not isinstance(expected, list) or not isinstance(generated, list):
        raise TypeError("BFCL expected and generated calls must be arrays")

    return ast_checker(
        functions,
        generated,
        expected,
        language,
        category,
        model_name,
    )


def evaluate_with_bfcl(
    row: object,
    output: str | None,
    *,
    model_name: str = "test",
    evaluator: Evaluator | None = None,
) -> Mapping[str, object]:
    """Return the official BFCL result for an Agent Management dataset row."""
    if not output:
        return {"valid": False, "error": ["empty output"]}
    try:
        expected = _decode_json(
            _field(row, "expected_output", _field(row, "expected"))
        )
        generated = _generated_output(output)
        if evaluator is not None:
            result = evaluator(expected, generated)
        else:
            metadata = _field(row, "metadata", {})
            if not isinstance(metadata, Mapping):
                return {"valid": False, "error": ["dataset row metadata is not a mapping"]}
            result = _official_result(
                expected,
                generated,
                metadata=metadata,
                model_name=model_name,
            )
    except (
        AssertionError,
        AttributeError,
        ImportError,
        KeyError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
    ) as error:
        return {"valid": False, "error": [str(error)]}
    return result


def score_with_bfcl(
    row: object,
    output: str | None,
    *,
    model_name: str = "test",
    evaluator: Evaluator | None = None,
) -> float:
    """Return BFCL's binary validity result for an Agent Management dataset row."""
    result = evaluate_with_bfcl(
        row,
        output,
        model_name=model_name,
        evaluator=evaluator,
    )
    return 1.0 if result.get("valid") is True else 0.0
