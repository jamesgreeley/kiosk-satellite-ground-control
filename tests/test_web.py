"""The web app's stylesheet, as a browser reads it."""

from __future__ import annotations

import re
from pathlib import Path

CSS = Path(__file__).resolve().parent.parent / "ground_control" / "web" / "app.css"


def _rules() -> tuple[list[str], list[int]]:
    """The selector of every top-level rule, and the lines with a closing brace that closes nothing."""
    text = re.sub(r"/\*.*?\*/", lambda m: "\n" * m.group().count("\n"), CSS.read_text(), flags=re.S)
    selectors, stray, depth, start, line = [], [], 0, 0, 1
    for i, ch in enumerate(text):
        if ch == "\n":
            line += 1
        elif ch == "{":
            if depth == 0:
                selectors.append(text[start:i].strip())
            depth += 1
        elif ch == "}":
            if depth == 0:
                stray.append(line)
            else:
                depth -= 1
            if depth == 0:
                start = i + 1
    assert depth == 0, "a block is never closed"
    return selectors, stray


def test_no_closing_brace_closes_nothing():
    """A browser reads a stray closing brace as the start of the next rule's
    selector and drops that rule. One before ``.toast`` left every short
    message as plain text under the footer (#14)."""
    assert _rules()[1] == []


def test_short_messages_float_over_the_page():
    selectors, _ = _rules()
    assert ".toast" in selectors
    toast = re.search(r"^\.toast \{([^}]*)\}", CSS.read_text(), flags=re.M).group(1)
    assert "position: fixed" in toast and "z-index" in toast
