from pathlib import Path

import pytest

from nb_expl.parser import (
    CellKind,
    NotebookFormatError,
    parse_file,
    parse_text,
)

EXAMPLE = Path(__file__).parent.parent / "examples" / "cash_application_engine.py"


@pytest.fixture(scope="module")
def nb():
    return parse_file(EXAMPLE)


# --- the example notebook -------------------------------------------------


def test_cell_count_matches_separators(nb):
    separators = sum(1 for line in nb.lines if line == "# COMMAND ----------")
    assert len(nb.cells) == separators + 1 == 47


def test_cells_are_ordered_and_do_not_overlap(nb):
    for prev, cur in zip(nb.cells, nb.cells[1:]):
        assert prev.end_line < cur.start_line
    assert [c.number for c in nb.cells] == list(range(1, 48))


def test_every_content_line_belongs_to_exactly_one_cell(nb):
    for n, text in enumerate(nb.lines, start=1):
        owners = [c for c in nb.cells if c.contains(n)]
        if n == 1 or text == "# COMMAND ----------":
            assert owners == []
        elif text.strip():
            assert len(owners) == 1, f"line {n} owned by {len(owners)} cells"


def test_raw_lines_match_file(nb):
    for c in nb.cells:
        assert list(c.raw_lines) == list(nb.lines[c.start_line - 1 : c.end_line])
        assert len(c.code_lines) == len(c.raw_lines)


def test_markdown_cell(nb):
    c = nb.cell(1)
    assert c.kind is CellKind.MARKDOWN
    assert (c.start_line, c.end_line) == (2, 4)
    assert c.magic == "%md"
    assert c.code_lines == ("", "## cash app v3", "new logic from Q2 - dont use v2 notebook anymore")


def test_run_cell_keeps_command(nb):
    c = nb.cell_for_line(44)
    assert c.kind is CellKind.RUN
    assert c.magic == "%run ../shared/fin_utils"
    assert c.code_line(44) == "%run ../shared/fin_utils"


def test_sql_cells(nb):
    sql = [c for c in nb.cells if c.kind is CellKind.SQL]
    assert [(c.start_line, c.end_line) for c in sql] == [(540, 541), (545, 547)]
    assert sql[0].code_line(541).startswith("select typ, count(*)")


def test_tol_assignments_are_in_different_cells(nb):
    first, second = nb.cell_for_line(33), nb.cell_for_line(361)
    assert first.code_line(33) == "TOL = 0.5"
    assert second.code_line(361) == "TOL = 5.0"
    assert first.number < second.number


def test_commented_out_cells(nb):
    commented = [(c.start_line, c.end_line) for c in nb.cells if c.commented_out]
    assert (148, 165) in commented  # old v2 matching
    assert (209, 230) in commented  # parked fuzzy matching
    assert not nb.cell_for_line(186).commented_out  # mixed cell, one line commented


def test_lookup_helpers(nb):
    assert nb.cell_for_line(1) is None  # header
    assert nb.cell_for_line(6) is None  # separator
    assert nb.snippet(33, 34) == [(33, "TOL = 0.5"), (34, "MAX_WO = 250.0")]
    with pytest.raises(IndexError):
        nb.line(nb.line_count + 1)
    with pytest.raises(IndexError):
        nb.cell(0)


# --- format edge cases ----------------------------------------------------


def test_missing_header_is_rejected():
    with pytest.raises(NotebookFormatError):
        parse_text("import os\n")
    with pytest.raises(NotebookFormatError):
        parse_text("")


def test_crlf_and_trailing_separator_whitespace():
    nb = parse_text(
        "# Databricks notebook source\r\nx = 1\r\n\r\n# COMMAND ----------  \r\n\r\ny = 2\r\n"
    )
    assert [(c.start_line, c.code) for c in nb.cells] == [(2, "x = 1"), (6, "y = 2")]


def test_empty_cell_keeps_numbering():
    nb = parse_text(
        "# Databricks notebook source\n"
        "a = 1\n"
        "# COMMAND ----------\n"
        "\n"
        "# COMMAND ----------\n"
        "b = 2\n"
    )
    assert [c.kind for c in nb.cells] == [CellKind.PYTHON, CellKind.EMPTY, CellKind.PYTHON]
    assert nb.cell(3).start_line == 6


def test_dbtitle_and_inline_magic():
    nb = parse_text(
        "# Databricks notebook source\n"
        "# DBTITLE 1,Load receipts\n"
        "r = spark.read.csv(p)\n"
        "# COMMAND ----------\n"
        "# MAGIC %md # Heading\n"
        "# MAGIC body\n"
        "# COMMAND ----------\n"
        "# MAGIC %pip install foo\n"
    )
    load, md, pip = nb.cells
    assert load.title == "Load receipts"
    assert load.code_lines == ("", "r = spark.read.csv(p)")
    assert not load.commented_out
    assert md.kind is CellKind.MARKDOWN and md.code_lines == ("# Heading", "body")
    assert pip.kind is CellKind.OTHER_MAGIC and pip.magic == "%pip"


def test_sql_notebook_source():
    nb = parse_text(
        "-- Databricks notebook source\n"
        "-- MAGIC %md\n"
        "-- MAGIC # Upgrade\n"
        "\n"
        "-- COMMAND ----------\n"
        "\n"
        "-- DBTITLE 1,Copy table\n"
        "CREATE TABLE t AS SELECT * FROM s;\n"
        "-- COMMAND ----------\n"
        "-- MAGIC %python\n"
        "-- MAGIC x = 1\n"
        "-- COMMAND ----------\n"
        "-- DROP TABLE t;\n"
    )
    assert nb.language == "sql"
    md, sql, py, old = nb.cells
    assert md.kind is CellKind.MARKDOWN and md.code_lines == ("", "# Upgrade")
    assert sql.kind is CellKind.SQL and sql.title == "Copy table" and sql.start_line == 7
    assert sql.code_line(8) == "CREATE TABLE t AS SELECT * FROM s;"
    assert py.kind is CellKind.PYTHON and py.code_lines == ("", "x = 1")
    assert old.kind is CellKind.SQL and old.commented_out


def test_scala_and_r_notebooks():
    scala = parse_text("// Databricks notebook source\nval x = 1\n// COMMAND ----------\n// MAGIC %sql\n// MAGIC SELECT 1\n")
    assert scala.language == "scala"
    assert [c.kind for c in scala.cells] == [CellKind.SCALA, CellKind.SQL]
    r = parse_text("# Databricks notebook source\nx <- 1\n", path="notebook.r")
    assert r.language == "r" and r.cells[0].kind is CellKind.R
    assert parse_text("# Databricks notebook source\nx = 1\n").cells[0].kind is CellKind.PYTHON


def test_separator_of_other_language_is_not_a_cell_break():
    nb = parse_text("# Databricks notebook source\nx = 1\n-- COMMAND ----------\ny = 2\n")
    assert len(nb.cells) == 1
