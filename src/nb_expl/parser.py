"""Parse Databricks notebook source files into cells with exact line numbers.

The parser is deliberately deterministic and LLM-free: every citation the app
produces is later checked against the structure built here, so line numbers
must match the file on disk exactly (1-based, as shown in any editor).

Databricks source format:

    # Databricks notebook source          <- line 1, mandatory header
    # MAGIC %md                           <- magic cells are prefixed "# MAGIC"
    # MAGIC Some markdown
    <blank>
    # COMMAND ----------                  <- cell separator
    <blank>
    # DBTITLE 1,Optional cell title
    x = 1                                 <- plain Python cell

SQL and Scala notebooks use the same layout with their own comment token
("-- Databricks notebook source", "-- COMMAND ----------", "-- MAGIC ..." and
"//" for Scala). R notebooks use "#" like Python and are recognised by their
.r extension. Plain cells take the notebook's language; magic cells say their own.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

HEADER = "# Databricks notebook source"
HEADER_RE = re.compile(r"^(#|--|//) Databricks notebook source$")
COMMENT_LANGUAGE = {"#": "python", "--": "sql", "//": "scala"}


class NotebookFormatError(ValueError):
    """Raised when a file is not a Databricks notebook source file."""


class CellKind(StrEnum):
    PYTHON = "python"
    MARKDOWN = "markdown"
    SQL = "sql"
    RUN = "run"  # %run: executes another notebook in this notebook's scope
    SHELL = "sh"
    SCALA = "scala"
    R = "r"
    OTHER_MAGIC = "magic"  # %pip, %fs, ...
    EMPTY = "empty"


_MAGIC_KINDS = {
    "%python": CellKind.PYTHON,
    "%md": CellKind.MARKDOWN,
    "%sql": CellKind.SQL,
    "%run": CellKind.RUN,
    "%sh": CellKind.SHELL,
    "%scala": CellKind.SCALA,
    "%r": CellKind.R,
}
_LANGUAGE_KIND = {"python": CellKind.PYTHON, "sql": CellKind.SQL, "scala": CellKind.SCALA, "r": CellKind.R}
_COMMENT = {CellKind.PYTHON: "#", CellKind.R: "#", CellKind.SQL: "--", CellKind.SCALA: "//"}


@dataclass(frozen=True)
class Cell:
    number: int
    """1-based cell number, counted the same way as in the Databricks UI."""
    kind: CellKind
    start_line: int
    """First non-blank line of the cell in the file (1-based, inclusive)."""
    end_line: int
    """Last non-blank line of the cell in the file (1-based, inclusive)."""
    raw_lines: tuple[str, ...]
    """File lines start_line..end_line exactly as written."""
    code_lines: tuple[str, ...]
    """Same length as raw_lines, with "# MAGIC" prefixes, the magic command and
    DBTITLE lines removed, so line i of code_lines is file line start_line + i."""
    title: str | None = None
    magic: str | None = None
    """The magic command, e.g. "%sql", or "%run ../shared/fin_utils"."""
    commented_out: bool = False
    """True for a code cell in which every non-blank line is a comment."""

    @property
    def code(self) -> str:
        return "\n".join(self.code_lines)

    def contains(self, line: int) -> bool:
        return self.start_line <= line <= self.end_line

    def code_line(self, line: int) -> str:
        """Code text for a 1-based file line inside this cell."""
        if not self.contains(line):
            raise IndexError(f"line {line} is outside cell {self.number}")
        return self.code_lines[line - self.start_line]


@dataclass(frozen=True)
class Notebook:
    path: str | None
    lines: tuple[str, ...]
    """All file lines; lines[0] is file line 1."""
    cells: tuple[Cell, ...] = field(default_factory=tuple)
    language: str = "python"
    """Default language of plain cells: python, sql, scala or r."""

    @property
    def line_count(self) -> int:
        return len(self.lines)

    def line(self, n: int) -> str:
        if not 1 <= n <= len(self.lines):
            raise IndexError(f"line {n} is outside 1..{len(self.lines)}")
        return self.lines[n - 1]

    def snippet(self, start: int, end: int) -> list[tuple[int, str]]:
        """Numbered file lines start..end (inclusive)."""
        if start > end:
            raise ValueError(f"start {start} > end {end}")
        return [(n, self.line(n)) for n in range(start, end + 1)]

    def cell(self, number: int) -> Cell:
        if not 1 <= number <= len(self.cells):
            raise IndexError(f"cell {number} is outside 1..{len(self.cells)}")
        return self.cells[number - 1]

    def cell_for_line(self, line: int) -> Cell | None:
        """The cell containing a file line, or None for header/separator/blank gaps."""
        for c in self.cells:
            if c.contains(line):
                return c
        return None


def parse_file(path: str | Path) -> Notebook:
    p = Path(path)
    # utf-8-sig strips a BOM if present; splitlines handles \n and \r\n.
    return parse_text(p.read_text(encoding="utf-8-sig"), path=str(p))


def parse_text(text: str, path: str | None = None) -> Notebook:
    """Parse notebook source text into cells.

    The header's comment token (#, -- or //) decides the separator and MAGIC patterns
    and the default language. Raises NotebookFormatError if the header is missing.
    """
    lines = tuple(text.splitlines())
    m = HEADER_RE.match(lines[0].strip()) if lines else None
    if not m:
        raise NotebookFormatError(
            f"not a Databricks notebook source file: first line must be {HEADER!r} "
            f"(or '-- Databricks notebook source' for SQL, '// …' for Scala)"
        )
    comment = m[1]
    language = COMMENT_LANGUAGE[comment]
    if comment == "#" and path and Path(path).suffix.lower() == ".r":
        language = "r"

    # Split into regions of (file_line_number, text) between separators.
    separator = f"{comment} COMMAND ----------"
    regions: list[list[tuple[int, str]]] = [[]]
    for n, line in enumerate(lines[1:], start=2):
        if line.rstrip() == separator:
            regions.append([])
        else:
            regions[-1].append((n, line))

    syntax = _Syntax(comment, _LANGUAGE_KIND[language])
    cells = tuple(_build_cell(i, region, syntax) for i, region in enumerate(regions, start=1))
    return Notebook(path=path, lines=lines, cells=cells, language=language)


class _Syntax:
    """Line patterns for one notebook's comment token."""

    def __init__(self, comment: str, default_kind: CellKind):
        c = re.escape(comment)
        self.magic = re.compile(rf"^{c} MAGIC ?(.*)$")
        self.dbtitle = re.compile(rf"^{c} DBTITLE \d+,(.*)$")
        self.default_kind = default_kind


def _build_cell(number: int, region: list[tuple[int, str]], syntax: _Syntax) -> Cell:
    """Build one cell from the (line number, text) pairs between two separators.

    Empty cells are kept so cell numbers match the Databricks UI. code_lines stays
    aligned with the file's lines, so line i of the cell is file line start_line + i.
    """
    body = [(n, t) for n, t in region if t.strip()]
    if not body:
        # Keep empty cells so numbering matches the Databricks UI.
        anchor = region[0][0] if region else 0
        return Cell(number, CellKind.EMPTY, anchor, anchor, (), ())

    start, end = body[0][0], body[-1][0]
    raw = [t for n, t in region if start <= n <= end]

    title = None
    code: list[str] = []
    for t in raw:
        m = syntax.dbtitle.match(t)
        if m and title is None and not code:
            title = m.group(1).strip()
            code.append("")
            continue
        m = syntax.magic.match(t)
        code.append(m.group(1) if m else t)

    is_magic = any(syntax.magic.match(t) for t in raw)
    kind, magic = syntax.default_kind, None
    if is_magic:
        kind, magic = _classify_magic(code, syntax.default_kind)

    comment = _COMMENT.get(kind)
    commented_out = comment is not None and all(
        not c.strip() or c.lstrip().startswith(comment) for c in code
    )

    return Cell(
        number=number,
        kind=kind,
        start_line=start,
        end_line=end,
        raw_lines=tuple(raw),
        code_lines=tuple(code),
        title=title,
        magic=magic,
        commented_out=commented_out,
    )


def _classify_magic(code: list[str], default_kind: CellKind) -> tuple[CellKind, str | None]:
    """Find the magic command, blank it out of code, and return the cell kind.

    For %run the command line *is* the content, so it is kept as code.
    """
    for i, c in enumerate(code):
        s = c.strip()
        if not s:
            continue
        if not s.startswith("%"):
            break
        command = s.split(maxsplit=1)[0]
        kind = _MAGIC_KINDS.get(command, CellKind.OTHER_MAGIC)
        if kind is not CellKind.RUN:
            # Keep anything after the command on the same line (e.g. "%md # Title").
            code[i] = s[len(command):].lstrip()
            return kind, command
        return kind, s
    return default_kind, None
