# Vixen Intelligence c.2026
"""Fixed-width table rendering for the terminal.

Every ``siar-app`` listing is a table, and the alternative is hand-counted format strings.
Those drift: a value outgrows its column, the header does not move with it, and the output
misaligns in exactly the cases — long slugs, long descriptions — where alignment is the only
thing making it readable.

This module is the one place that measures. It is stdlib only: this is a pip-installable tool
whose value is partly in what it does *not* drag in, and a table renderer does not justify a
dependency on ``rich`` or ``tabulate``. Copied verbatim from ``../siar``; keep the two in step.

Contract:

* Every column is sized to its widest cell, header included. Nothing is truncated silently — a
  clipped slug is unrecoverable, and a table that overflows a narrow terminal is not.
* One column may be nominated as ``flex``. It absorbs whatever width is left over and **wraps**
  onto continuation lines instead of pushing the table past the right edge. That is what makes a
  description column readable at 80 columns and generous at 200.
* The renderer returns a string and prints nothing. Callers decide where it goes, which is what
  keeps it testable without capturing stdout.
"""
from __future__ import annotations

import shutil
import textwrap
from collections.abc import Sequence

__all__ = ["MIN_FLEX_WIDTH", "render_table", "terminal_width"]

#: A flex column is never squeezed below this. Past roughly this point wrapping produces a column
#: of two-word fragments that is harder to read than an overflowing line.
MIN_FLEX_WIDTH = 28


def terminal_width(default: int = 100) -> int:
    """The usable terminal width in columns.

    Args:
        default: Fallback when the output is not a terminal — a pipe, a file, a CI log. Chosen
            wide enough that a redirected table is not needlessly cramped.

    Returns:
        The column count, never less than 40.
    """
    return max(40, shutil.get_terminal_size((default, 24)).columns)


def render_table(
    headers: Sequence[str],
    rows: Sequence[Sequence[object]],
    *,
    flex: int | None = None,
    align: Sequence[str] | None = None,
    width: int | None = None,
    gap: int = 2,
    rule: bool = True,
) -> str:
    """Render rows as an aligned, optionally wrapping, plain-text table.

    Args:
        headers: Column headings. Rendered as given — callers upper-case them, matching the rest
            of the CLI.
        rows: One sequence of cells per row. Cells are stringified with ``str``; ``None`` becomes
            an empty cell rather than the word "None".
        flex: Index of the column allowed to wrap and to soak up the leftover width. ``None``
            (the default) sizes every column to its content and lets the line run long.
        align: Per-column alignment, ``"<"`` or ``">"``. Defaults to left for every column. Short
            sequences are padded with ``"<"``.
        width: Total width to fit within. Defaults to :func:`terminal_width`. Only consulted when
            ``flex`` is set.
        gap: Spaces between columns.
        rule: Draw a dashed rule under the header row.

    Returns:
        The table as a multi-line string, no trailing newline. Empty string if ``headers`` is
        empty. Every line is right-stripped, so an empty trailing cell costs no whitespace.

    Raises:
        ValueError: If a row's cell count does not match the header count, or ``flex`` is out of
            range. Both are programming errors in the caller and are worth failing loudly.
    """
    if not headers:
        return ""
    n_cols = len(headers)
    if flex is not None and not 0 <= flex < n_cols:
        raise ValueError(f"flex column {flex} out of range for {n_cols} columns")

    cells: list[list[str]] = []
    for i, row in enumerate(rows):
        row = list(row)
        if len(row) != n_cols:
            raise ValueError(f"row {i} has {len(row)} cells, expected {n_cols}")
        cells.append(["" if c is None else str(c) for c in row])

    aligns = list(align or [])
    aligns += ["<"] * (n_cols - len(aligns))

    natural = [
        max(len(headers[c]), max((len(r[c]) for r in cells), default=0)) for c in range(n_cols)
    ]

    widths = list(natural)
    if flex is not None:
        total = width if width is not None else terminal_width()
        fixed = sum(w for c, w in enumerate(widths) if c != flex) + gap * (n_cols - 1)
        widths[flex] = max(MIN_FLEX_WIDTH, min(natural[flex], total - fixed))

    pad = " " * gap

    def lay(parts: Sequence[str]) -> str:
        return pad.join(f"{p:{aligns[c]}{widths[c]}}" for c, p in enumerate(parts)).rstrip()

    out = [lay(headers)]
    if rule:
        out.append(pad.join("-" * w for w in widths).rstrip())

    for row in cells:
        if flex is None:
            out.append(lay(row))
            continue
        # The flex cell may become several lines; every other column is blank on the
        # continuations, so the eye tracks one record down the page.
        chunks = textwrap.wrap(row[flex], widths[flex]) or [""]
        for line_no, chunk in enumerate(chunks):
            parts = list(row) if line_no == 0 else [""] * n_cols
            parts[flex] = chunk
            out.append(lay(parts))
    return "\n".join(out)
