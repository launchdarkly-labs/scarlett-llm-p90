"""Export a deterministic, disjoint BFCL multi-turn pilot and confirmatory set."""

from __future__ import annotations

import argparse
import hashlib
import json
from importlib.resources import files
from pathlib import Path

DATASET = "BFCL_v4_multi_turn_base.json"
ANSWERS = f"possible_answer/{DATASET}"


def read_jsonl(path: object) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def render_input(question: object) -> str:
    if not isinstance(question, list):
        raise TypeError("BFCL question must be an array of turns")
    turns = []
    for index, messages in enumerate(question, start=1):
        if not isinstance(messages, list):
            raise TypeError("BFCL turn must be an array of messages")
        content = "\n".join(
            str(message.get("content", ""))
            for message in messages
            if isinstance(message, dict)
        )
        turns.append(f"Turn {index}:\n{content}")
    return "\n\n".join(turns)


def convert(entry: dict[str, object], answer: dict[str, object]) -> dict[str, object]:
    source_id = str(entry["id"])
    category = source_id.rsplit("_", 1)[0]
    return {
        "input": render_input(entry["question"]),
        "expected_output": json.dumps(answer["ground_truth"], separators=(",", ":")),
        "variables": {},
        "metadata": {
            "source_id": source_id,
            "test_category": category,
            "test_entry": {
                "id": source_id,
                "question": entry["question"],
                "initial_config": entry["initial_config"],
                "involved_classes": entry["involved_classes"],
            },
        },
    }


def stable_order(row: dict[str, object]) -> str:
    metadata = row["metadata"]
    assert isinstance(metadata, dict)
    return hashlib.sha256(str(metadata["source_id"]).encode()).hexdigest()


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot-size", type=int, default=30)
    parser.add_argument("--out-dir", type=Path, default=Path("data"))
    args = parser.parse_args()

    data_root = files("bfcl_eval").joinpath("data")
    entries = read_jsonl(data_root.joinpath(DATASET))
    answers = {
        str(row["id"]): row for row in read_jsonl(data_root.joinpath(ANSWERS))
    }
    rows = sorted(
        (convert(entry, answers[str(entry["id"])]) for entry in entries),
        key=stable_order,
    )
    if not 0 < args.pilot_size < len(rows):
        raise ValueError("pilot-size must leave at least one confirmatory row")

    write_jsonl(args.out_dir / "bfcl_pilot.jsonl", rows[: args.pilot_size])
    write_jsonl(args.out_dir / "bfcl_confirmatory.jsonl", rows[args.pilot_size :])


if __name__ == "__main__":
    main()
