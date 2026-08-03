"""Markdown table formatting for the result reports."""
from __future__ import annotations


def format_table(header: list[str], rows: list[list]) -> str:
    """GitHub-flavored markdown table; cells are rendered with ``str()``."""
    lines = ["| " + " | ".join(str(h) for h in header) + " |",
             "|" + "|".join("---" for _ in header) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(lines)
