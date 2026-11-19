from pathlib import Path

import pytest

from nb_expl.analysis import FactKind, analyze
from nb_expl.parser import parse_file, parse_text
from nb_expl.refs import Ref, check_ref, find_refs, parse_ref

EXAMPLE = Path(__file__).parent.parent / "examples" / "cash_application_engine.py"


@pytest.fixture(scope="module")
def nb():
    return parse_file(EXAMPLE)


@pytest.fixture(scope="module")
def facts(nb):
    return analyze(nb)


def by_line(facts, kind):
    return {f.ref.start: f for f in facts.of(kind)}


# --- refs -----------------------------------------------------------------


def test_ref_roundtrip():
    assert str(Ref(4, 33, 33)) == "C4:L33"
    assert str(Ref(26, 286, 300)) == "C26:L286-300"
    assert parse_ref("[C26:L286-300]") == Ref(26, 286, 300)
    assert parse_ref("C26:L286-L300") == Ref(26, 286, 300)
    assert find_refs("see [C4:L33] and C29:L361.") == [Ref(4, 33, 33), Ref(29, 361, 361)]
    with pytest.raises(ValueError):
        parse_ref("line 33")


def test_check_ref(nb):
    assert check_ref(Ref(4, 33, 33), nb) is None
    assert "outside cell" in check_ref(Ref(4, 33, 48), nb)
    assert "does not exist" in check_ref(Ref(99, 1, 1), nb)


def test_every_fact_ref_is_valid(nb, facts):
    for f in facts.facts:
        assert check_ref(f.ref, nb) is None, f


# --- the facts that matter in the example -----------------------------------


def test_tol_reassignment_and_effective_values(facts):
    (re,) = facts.of(FactKind.CONSTANT_REASSIGNED)
    assert re.ref == Ref(29, 361, 361)
    assert re.data["name"] == "TOL" and re.data["first_line"] == 33

    eff = by_line(facts, FactKind.EFFECTIVE_VALUE)
    assert eff[375].data["assigned_line"] == 361  # if rem > TOL
    assert eff[377].data["assigned_line"] == 361  # do_wo() reads global TOL
    assert "do_wo()" in eff[377].summary


def test_constants(facts):
    consts = {f.data["name"]: f.data["value"] for f in facts.of(FactKind.CONSTANT) if f.ref.start < 100}
    assert consts["TOL"] == 0.5 and consts["MAX_WO"] == 250.0 and consts["EX_TOL"] == 0.01
    assert consts["EXCL"] == ["C10077", "C10412"]


def test_widgets(facts):
    w = {f.data["name"]: f.data for f in facts.of(FactKind.WIDGET)}
    assert set(w) == {"src", "run_dt", "co", "dry"}
    assert w["dry"]["default"] == "Y" and w["dry"]["choices"] == ["Y", "N"]


def test_csv_reads(facts):
    reads = facts.of(FactKind.FILE_READ)
    assert [f.data["assigned_to"] for f in reads] == [["r"], ["oi"], ["cm"], ["ov"], ["fx"]]
    assert all(f.data["options"] == {"header": True} for f in reads)


def test_table_writes_are_guarded_by_dry(facts):
    writes = {f.data["table"]: f.data for f in facts.of(FactKind.TABLE_WRITE)}
    assert writes["fin_cash.alloc_lines"]["mode"] == "overwrite"  # resolved via tgt
    assert writes["fin_cash.open_items_after"]["mode"] == "overwrite"
    assert writes["fin_cash.ar_aging_snap"]["mode"] == "append"
    assert all(w["guards"] == ["dry == 'N'"] for w in writes.values())


def test_notifications_run_even_in_dry_run(facts):
    effects = {f.data["call"]: f.data for f in facts.of(FactKind.SIDE_EFFECT)}
    assert effects["external function notify_ar"]["guards"] == []
    assert effects["external function log_run"]["guards"] == []


def test_external_names_from_run(facts):
    ext = {f.data["name"]: f.data["origin"] for f in facts.of(FactKind.EXTERNAL_NAME)}
    assert ext["get_gl_map"] == ext["notify_ar"] == ext["log_run"] == "run"
    assert ext["spark"] == ext["dbutils"] == ext["display"] == "databricks"
    assert "StringType" not in ext  # provided by the star import
    (run,) = facts.of(FactKind.RUN_DEPENDENCY)
    assert run.data["notebook"] == "../shared/fin_utils"


def test_functions(facts):
    fns = {f.data["name"]: f.data for f in facts.of(FactKind.FUNCTION)}
    assert fns["apply_ref"]["global_mutations"] == ["bal"]
    assert fns["apply_ref"]["arg_mutations"] == ["lines"]
    assert "TOL" in fns["do_wo"]["global_reads"]
    assert "EX_TOL" in fns["apply_exact"]["global_reads"]
    assert fns["alloc_by_aging"]["used_on"] == []


def test_unused(facts):
    assert {f.data["name"] for f in facts.of(FactKind.UNUSED_NAME)} == {"MAX_WO", "alloc_by_aging"}
    assert {f.data["name"] for f in facts.of(FactKind.UNUSED_IMPORT)} == {"math", "reduce"}


def test_commented_code(facts):
    refs = {str(f.ref) for f in facts.of(FactKind.COMMENTED_CODE)}
    assert {"C15:L148-165", "C20:L209-230", "C18:L186", "C28:L351-352", "C46:L546"} <= refs
    # plain prose comments are not code
    assert not any(f.ref.start == 230 and f.ref.end == 230 for f in facts.of(FactKind.COMMENTED_CODE))


def test_thresholds(facts):
    t = by_line(facts, FactKind.THRESHOLD)
    assert t[322].data["function"] == "apply_fifo"
    assert t[492].data["values"] == [10000]


def test_dataflow(facts):
    flow = by_line(facts, FactKind.DATAFLOW)
    assert flow[136].data["uses"] == ["ov", "r", "x1", "x2"]
    assert flow[271].data["target"] == "rp"
    assert flow[411].data["uses"] == ["get_gl_map"]


def test_render_contains_refs(facts):
    text = facts.render()
    assert "## constant_reassigned" in text
    assert "[C29:L361] TOL is reassigned" in text


# --- synthetic edge cases ---------------------------------------------------


def test_syntax_error_cell_is_reported_not_fatal():
    nb = parse_text("# Databricks notebook source\nx = (\n# COMMAND ----------\ny = 1\n")
    facts = analyze(nb)
    (err,) = facts.of(FactKind.PARSE_ERROR)
    assert err.ref.cell == 1


def test_else_branch_guard():
    nb = parse_text(
        "# Databricks notebook source\n"
        "if dry == 'N':\n"
        "    pass\n"
        "else:\n"
        "    df.write.mode('append').saveAsTable('t')\n"
    )
    (w,) = analyze(nb).of(FactKind.TABLE_WRITE)
    assert w.data["guards"] == ["not (dry == 'N')"]


def test_ipython_lines_are_side_effects_not_parse_errors():
    nb = parse_text(
        "# Databricks notebook source\n"
        "!wget -q https://example.com/data.csv -O /tmp/data.csv\n"
        "%pip install foo\n"
        "df = spark.read.csv('/tmp/data.csv')\n"
        "if df.count() > 0:\n"
        "    !rm /tmp/data.csv\n"
    )
    facts = analyze(nb)
    assert not facts.of(FactKind.PARSE_ERROR)
    effects = {f.ref.start: f.summary for f in facts.of(FactKind.SIDE_EFFECT)}
    assert effects[2].startswith("Runs shell command: !wget")
    assert effects[3].startswith("Runs %pip magic")
    assert 6 in effects
    assert facts.of(FactKind.FILE_READ)


def test_valid_python_is_not_rewritten():
    nb = parse_text("# Databricks notebook source\nx = (10\n     % 3)\ny = x != 1\n")
    facts = analyze(nb)
    assert not facts.of(FactKind.PARSE_ERROR)
    assert not facts.of(FactKind.SIDE_EFFECT)


def test_sql_tables_patterns():
    from nb_expl.analysis import sql_tables

    assert sql_tables("DROP TABLE IF EXISTS db.t") == ([("DROP TABLE", "db.t")], [])
    assert sql_tables("CREATE TABLE IF NOT EXISTS {db}.x LIKE src COPY LOCATION") == (
        [("CREATE TABLE", "{db}.x")], [])
    assert sql_tables("DELETE FROM t WHERE id IN (SELECT id FROM s)") == ([("DELETE FROM", "t")], ["s"])
    assert sql_tables("CREATE OR REPLACE TEMP VIEW v AS SELECT * FROM a JOIN b") == (
        [("CREATE OR REPLACE TEMP VIEW", "v")], ["a", "b"])
    assert sql_tables("DROP DATABASE IF EXISTS `main`.`db` CASCADE") == (
        [("DROP DATABASE", "`main`.`db`")], [])


def test_spark_sql_statements_are_classified():
    nb = parse_text(
        "# Databricks notebook source\n"
        "db = 'sales'\n"
        "if dry == 'N':\n"
        "    spark.sql(f'DROP DATABASE IF EXISTS {db} CASCADE')\n"
        "df = spark.sql('SELECT * FROM orders o JOIN customers c ON o.c = c.id')\n"
        "spark.sql(f'USE DATABASE {db}')\n"
        "q = 'INSERT INTO audit SELECT * FROM staging'\n"
        "spark.sql(q)\n"
        "spark.sql(build_query())\n"
    )
    facts = analyze(nb)
    writes = {f.ref.start: f for f in facts.of(FactKind.TABLE_WRITE)}
    assert writes[4].data["table"] == "{db}" and writes[4].data["guards"] == ["dry == 'N'"]
    assert writes[8].data["table"] == "audit"
    reads = {(f.ref.start, f.data["source"]) for f in facts.of(FactKind.TABLE_READ)}
    assert reads == {(5, "orders"), (5, "customers"), (8, "staging")}
    effects = [(f.ref.start, f.summary) for f in facts.of(FactKind.SIDE_EFFECT)]
    assert any(n == 6 and "USE DATABASE {db}" in s for n, s in effects)
    assert any(n == 9 and "not known statically" in s for n, s in effects)


SCALA_NOTEBOOK = (
    "// Databricks notebook source\n"                                   # 1
    'val db = "sales"\n'                                                 # 2
    'val orders = spark.table("raw.orders")\n'                           # 3
    "// COMMAND ----------\n"                                            # 4
    'spark.sql(s"DROP TABLE IF EXISTS $db.daily")\n'                     # 5
    'spark.sql("""\n'                                                    # 6
    "  INSERT INTO ${db}.daily\n"                                        # 7
    "  SELECT * FROM raw.orders o JOIN raw.customers c ON o.c = c.id\n"  # 8
    '""")\n'                                                             # 9
    'orders.write.mode("overwrite").saveAsTable("sales.orders_copy")\n'  # 10
    '// spark.sql("DROP DATABASE prod")\n'                               # 11
    "spark.sql(buildQuery())\n"                                          # 12
)


def test_scala_cells_table_access():
    facts = analyze(parse_text(SCALA_NOTEBOOK))
    writes = {(f.ref.start, f.ref.end, f.data["table"]) for f in facts.of(FactKind.TABLE_WRITE)}
    assert writes == {(5, 5, "{db}.daily"), (6, 9, "${db}.daily"), (10, 10, "sales.orders_copy")}
    reads = {(f.ref.start, f.data["source"]) for f in facts.of(FactKind.TABLE_READ)}
    assert reads == {(3, "raw.orders"), (6, "raw.orders"), (6, "raw.customers")}
    assert all("not analysed" in f.summary for f in facts.of(FactKind.TABLE_WRITE))
    (dynamic,) = facts.of(FactKind.SIDE_EFFECT)
    assert dynamic.ref.start == 12 and "not known statically" in dynamic.summary


def test_scala_citations_point_at_the_code():
    nb = parse_text(SCALA_NOTEBOOK)
    for f in analyze(nb).facts:
        assert check_ref(f.ref, nb) is None
