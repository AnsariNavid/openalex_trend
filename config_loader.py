#!/usr/bin/env python3
"""Small helpers for robust JSON config loading."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


def _strip_json_comments(text: str) -> str:
    text = re.sub(r"//.*?$", "", text, flags=re.MULTILINE)
    text = re.sub(r"#.*?$", "", text, flags=re.MULTILINE)
    return text


def _strip_trailing_commas(text: str) -> str:
    return re.sub(r",\s*([}\]])", r"\1", text)


def load_json_file(path: str | Path, *, tolerate_comments: bool = False, tolerate_trailing_commas: bool = True) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p}")

    raw_text = p.read_text(encoding="utf-8")
    text = raw_text
    if tolerate_comments:
        text = _strip_json_comments(text)
    if tolerate_trailing_commas:
        text = _strip_trailing_commas(text)

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Invalid JSON in {p} at line {e.lineno}, column {e.colno}: {e.msg}. "
            "Tip: check for missing commas/quotes around this location."
        ) from e

    if not isinstance(parsed, dict):
        raise ValueError(f"Config in {p} must be a JSON object.")
    return parsed
