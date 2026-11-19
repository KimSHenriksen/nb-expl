"""Citations ("refs") pointing at a cell and a file line range.

Text form: ``C17:L57`` or ``C17:L57-60`` (cell number, 1-based file lines).
File line numbers are used, rather than lines within the cell, so a ref can be
opened directly in an editor; the cell number matches the Databricks UI.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .parser import Notebook

REF_RE = re.compile(r"C(\d+):L(\d+)(?:-L?(\d+))?")


@dataclass(frozen=True, order=True)
class Ref:
    cell: int
    start: int
    end: int

    def __post_init__(self) -> None:
        """Reject inverted ranges such as C3:L10-5."""
        if self.start > self.end:
            raise ValueError(f"ref start {self.start} > end {self.end}")

    def __str__(self) -> str:
        if self.start == self.end:
            return f"C{self.cell}:L{self.start}"
        return f"C{self.cell}:L{self.start}-{self.end}"


def parse_ref(text: str) -> Ref:
    """Parse "C26:L320" or "C26:L286-300", with or without [brackets]; raises ValueError otherwise."""
    m = REF_RE.fullmatch(text.strip().strip("[]"))
    if not m:
        raise ValueError(f"not a ref: {text!r}")
    cell, start, end = int(m[1]), int(m[2]), int(m[3] or m[2])
    return Ref(cell, start, end)


def find_refs(text: str) -> list[Ref]:
    """All refs mentioned anywhere in a piece of text."""
    out = []
    for m in REF_RE.finditer(text):
        try:
            out.append(Ref(int(m[1]), int(m[2]), int(m[3] or m[2])))
        except ValueError:
            continue
    return out


def check_ref(ref: Ref, nb: Notebook) -> str | None:
    """Return an error message if the ref does not point inside its cell, else None."""
    if not 1 <= ref.cell <= len(nb.cells):
        return f"{ref}: cell {ref.cell} does not exist (1..{len(nb.cells)})"
    cell = nb.cell(ref.cell)
    if not (cell.contains(ref.start) and cell.contains(ref.end)):
        return (
            f"{ref}: lines {ref.start}-{ref.end} are outside cell {ref.cell} "
            f"(lines {cell.start_line}-{cell.end_line})"
        )
    return None
