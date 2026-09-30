"""Parsing and serialization helpers for model decisions."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

_DECISION_PATTERN = re.compile(r"\b(admit|reject)\b", re.IGNORECASE)
_BANNED_EXPLANATION_TEXT = ("[your explanation here]", "example")


def normalize_decision(value: Any) -> str | None:
    """Return the first explicit ``admit``/``reject`` label, if present."""
    if value is None:
        return None
    match = _DECISION_PATTERN.search(str(value))
    return match.group(1).lower() if match else None


def _json_objects(text: str) -> Iterable[dict[str, Any]]:
    """Yield JSON objects embedded in text, respecting strings and escapes."""
    start: int | None = None
    depth = 0
    in_string = False
    escaped = False
    for index, character in enumerate(text):
        if start is None:
            if character == "{":
                start = index
                depth = 1
                in_string = False
                escaped = False
            continue

        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            continue
        if character == '"':
            in_string = True
        elif character == "{":
            depth += 1
        elif character == "}":
            depth -= 1
            if depth == 0:
                candidate = text[start : index + 1]
                try:
                    parsed = json.loads(candidate)
                except json.JSONDecodeError:
                    pass
                else:
                    if isinstance(parsed, dict):
                        yield parsed
                start = None


def parse_system2_response(raw_response: Any) -> dict[str, Any]:
    """Parse a chain-of-thought response without discarding malformed output."""
    raw = "" if raw_response is None else str(raw_response).strip()
    for parsed in _json_objects(raw):
        if set(parsed) != {"EXPLANATION", "DECISION"}:
            continue
        explanation = parsed.get("EXPLANATION")
        decision = normalize_decision(parsed.get("DECISION"))
        if not isinstance(explanation, str) or decision is None:
            continue
        if any(text in explanation.lower() for text in _BANNED_EXPLANATION_TEXT):
            continue
        return {
            "decision": decision,
            "explanation": explanation.strip(),
            "parse_status": "valid_json",
        }
    return {
        "decision": normalize_decision(raw),
        "explanation": None,
        "parse_status": "fallback_label" if normalize_decision(raw) else "unparsed",
    }


def parse_response(raw_response: Any, *, mode: str) -> dict[str, Any]:
    """Parse a response from either experimental reasoning condition."""
    if mode == "system2":
        return parse_system2_response(raw_response)
    if mode != "system1":
        raise ValueError("mode must be 'system1' or 'system2'")
    decision = normalize_decision(raw_response)
    return {
        "decision": decision,
        "explanation": None,
        "parse_status": "label" if decision else "unparsed",
    }


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read UTF-8 JSON Lines records."""
    records: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Line {line_number} is not a JSON object")
            records.append(value)
    return records


def write_jsonl(records: Iterable[Mapping[str, Any]], path: str | Path) -> None:
    """Write records atomically as UTF-8 JSON Lines."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(dict(record), ensure_ascii=False) + "\n")
    temporary.replace(destination)
