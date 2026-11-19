"""Deterministic static analysis of a notebook into citable facts.

Everything here is computed from the AST (Python cells) or simple patterns
(SQL cells, comments), without an LLM. The resulting fact sheet is given to
the LLM as ground truth, and each fact already carries its ref, so the facts
that matter most for safety (reassigned constants, unguarded side effects,
table writes, hidden dependencies) are never left to the model to notice.

Assumption: the notebook is executed top to bottom, once, so file order equals
execution order. Facts that depend on this say so in their summary.
"""

from __future__ import annotations

import ast
import builtins
import re
from collections import defaultdict
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Iterator

from .parser import Cell, CellKind, Notebook
from .refs import Ref

DATABRICKS_GLOBALS = {
    "spark", "dbutils", "display", "displayHTML", "sc", "sqlContext", "getArgument",
}
BUILTINS = set(dir(builtins))
# Names a star import is known to provide (heuristic, extend as needed).
STAR_EXPORTS = {
    "pyspark.sql.types": lambda n: n.endswith("Type") or n in {"StructField", "Row"},
    "pyspark.sql.functions": lambda n: n.islower() or n == "PandasUDFType",
}
MUTATING_METHODS = {
    "append", "extend", "insert", "pop", "remove", "clear", "update", "setdefault",
    "sort", "add", "discard",
}
READ_METHODS = {"csv", "parquet", "json", "orc", "text", "load", "table", "jdbc"}
WRITE_METHODS = {"saveAsTable", "insertInto", "save", "csv", "parquet", "json", "orc", "text"}
# A table name, also when templated: db.t, `db`.`t`, {catalog}.{db}.t, ${widget}.t
_SQL_NAME = r"(?P<name>[A-Za-z_{`$][\w.{}`$]*)"
SQL_TABLE_RE = re.compile(rf"\b(?:from|join)\s+{_SQL_NAME}", re.IGNORECASE)
# IPython-style lines that Databricks runs but Python cannot parse: "!cmd" and "%magic".
_IPYTHON_LINE_RE = re.compile(r"^\s*(![^=]|%[A-Za-z])")
SQL_WRITE_RE = re.compile(
    r"\b(?P<op>insert\s+(?:into|overwrite)(?:\s+table)?|merge\s+into|update|delete\s+from|"
    r"(?:create(?:\s+or\s+replace)?|drop|alter|truncate)(?:\s+(?:global\s+)?temp(?:orary)?)?"
    r"\s+(?:table|view|database|schema|catalog))"
    rf"(?:\s+if\s+(?:not\s+)?exists)?\s+{_SQL_NAME}",
    re.IGNORECASE,
)
_SQL_READ_ONLY = {"SELECT", "WITH", "DESCRIBE", "DESC", "SHOW", "EXPLAIN"}

# Scala cells get no AST; these patterns find table access by text instead.
_SCALA_STRING = r'(?P<lit>[sf]?"""[\s\S]*?"""|[sf]?"(?:[^"\\\n]|\\.)*")'
_SCALA_SQL_RE = re.compile(rf"\bspark\.sql\(\s*(?:{_SCALA_STRING})?")
_SCALA_WRITE_RE = re.compile(r'\.(?P<method>saveAsTable|insertInto)\(\s*"(?P<name>[^"]+)"')
_SCALA_READ_RE = re.compile(r'\bspark\.(?:read\.)?table\(\s*"(?P<name>[^"]+)"')
_SCALA_INTERPOLATION_RE = re.compile(r"\$\{([^}]*)\}|\$([A-Za-z_]\w*)")
SQL_CODE_RE = re.compile(r"^\s*(select|insert|update|delete|merge|create|drop|with)\b", re.I)


class FactKind(StrEnum):
    WIDGET = "widget"
    CONSTANT = "constant"
    CONSTANT_REASSIGNED = "constant_reassigned"
    EFFECTIVE_VALUE = "effective_value"
    FILE_READ = "file_read"
    TABLE_READ = "table_read"
    TABLE_WRITE = "table_write"
    SIDE_EFFECT = "side_effect"
    RUN_DEPENDENCY = "run_dependency"
    EXTERNAL_NAME = "external_name"
    FUNCTION = "function"
    UNUSED_NAME = "unused_name"
    UNUSED_IMPORT = "unused_import"
    COMMENTED_CODE = "commented_code"
    THRESHOLD = "threshold"
    DATAFLOW = "dataflow"
    PARSE_ERROR = "parse_error"


@dataclass(frozen=True)
class Fact:
    kind: FactKind
    ref: Ref
    summary: str
    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"kind": str(self.kind), "ref": str(self.ref), "summary": self.summary, "data": self.data}


@dataclass
class FactSheet:
    facts: list[Fact]

    def of(self, kind: FactKind) -> list[Fact]:
        return [f for f in self.facts if f.kind is kind]

    def render(self) -> str:
        """Compact plain-text form for LLM prompts, grouped by kind."""
        out = []
        for kind in FactKind:
            facts = self.of(kind)
            if not facts:
                continue
            out.append(f"## {kind}")
            out.extend(f"[{f.ref}] {f.summary}" for f in facts)
            out.append("")
        return "\n".join(out).rstrip() + "\n"

    def to_dict(self) -> list[dict[str, Any]]:
        return [f.to_dict() for f in self.facts]


# --- internal model ---------------------------------------------------------


@dataclass
class _Store:
    name: str
    kind: str  # assign | augassign | mutate | loop | function | import
    ref: Ref
    stmt: ast.AST
    value: ast.AST | None = None


@dataclass
class _Load:
    name: str
    ref: Ref
    node: ast.Name
    function: str | None  # enclosing module-level function, if any


@dataclass
class _Function:
    name: str
    ref: Ref
    node: ast.FunctionDef
    params: list[str]
    locals: set[str]
    global_reads: set[str] = field(default_factory=set)
    global_mutations: set[str] = field(default_factory=set)
    arg_mutations: set[str] = field(default_factory=set)
    calls: set[str] = field(default_factory=set)


class _Analyzer:
    def __init__(self, nb: Notebook):
        self.nb = nb
        self.facts: list[Fact] = []
        self.trees: list[tuple[Cell, ast.Module]] = []
        self.stores: dict[str, list[_Store]] = defaultdict(list)
        self.loads: list[_Load] = []
        self.all_bound: set[str] = set()
        self.functions: dict[str, _Function] = {}
        self.star_imports: list[str] = []

    # -- helpers --

    def ref(self, cell: Cell, node: ast.AST) -> Ref:
        """Ref for an AST node, translating cell-relative line numbers to file lines."""
        start = cell.start_line + node.lineno - 1
        end = cell.start_line + (node.end_lineno or node.lineno) - 1
        return Ref(cell.number, start, end)

    def add(self, kind: FactKind, ref: Ref, summary: str, **data: Any) -> None:
        """Record one fact; every fact producer goes through here."""
        self.facts.append(Fact(kind, ref, summary, data))

    def literal(self, name: str, before_line: int | None = None) -> Any:
        """Literal value of the latest module-level literal assignment of name."""
        val = None
        for s in self.stores.get(name, []):
            if s.kind != "assign" or s.value is None:
                continue
            if before_line is not None and s.ref.start >= before_line:
                continue
            ok, v = _literal(s.value)
            if ok:
                val = v
        return val

    # -- driver --

    def run(self) -> FactSheet:
        """Parse every Python cell, collect names and functions, then run the fact producers.

        The order matters: collection must finish before the producers read it, and
        dataflow runs last because it uses the external names found before it.
        """
        for cell in self.nb.cells:
            if cell.kind is CellKind.PYTHON and not cell.commented_out:
                tree = self._parse_cell(cell)
                if tree is not None:
                    _annotate_parents(tree)
                    self.trees.append((cell, tree))

        for cell, tree in self.trees:
            self.collect_scope(cell, tree)
        for cell, tree in self.trees:
            self.collect_functions(cell, tree)

        self.widgets()
        self.constants()
        self.reads_and_writes()
        self.run_cells_and_sql()
        self.scala_cells()
        self.external_names()
        self.function_facts()
        self.unused()
        self.commented_code()
        self.thresholds()
        self.dataflow()
        self.facts.sort(key=lambda f: (f.ref.start, f.ref.end, list(FactKind).index(f.kind)))
        return FactSheet(self.facts)

    def _parse_cell(self, cell: Cell) -> ast.Module | None:
        """Parse a Python cell. IPython-style lines ("!wget ...", "%pip install ...")
        are not Python; if the cell only parses without them, they are blanked out
        (keeping line numbers) and reported as side effects."""
        try:
            return ast.parse(cell.code)
        except SyntaxError as first_error:
            error = first_error
        shell = [(i, line) for i, line in enumerate(cell.code_lines) if _IPYTHON_LINE_RE.match(line)]
        if shell:
            lines = list(cell.code_lines)
            for i, line in shell:
                indent = line[: len(line) - len(line.lstrip())]
                lines[i] = indent + "pass"
            try:
                tree = ast.parse("\n".join(lines))
            except SyntaxError as e:
                error = e
            else:
                for i, line in shell:
                    n = cell.start_line + i
                    command = line.strip()
                    what = "shell command" if command.startswith("!") else f"{command.split()[0]} magic"
                    self.add(FactKind.SIDE_EFFECT, Ref(cell.number, n, n),
                             f"Runs {what}: {command[:100]}", call=command.split()[0], guards=[])
                return tree
        line = cell.start_line + (error.lineno or 1) - 1
        self.add(FactKind.PARSE_ERROR, Ref(cell.number, line, line),
                 f"Python syntax error, cell not analysed: {error.msg}")
        return None

    # -- collection --

    def collect_scope(self, cell: Cell, tree: ast.Module) -> None:
        """Record module-level assignments, imports and definitions (stores) and every name
        loaded anywhere in the cell (loads), across all cells, for the cross-cell facts.
        """
        for stmt in _module_statements(tree.body):
            ref = self.ref(cell, stmt)
            if isinstance(stmt, ast.Assign):
                for t in stmt.targets:
                    self._store_target(t, "assign", ref, stmt, stmt.value)
            elif isinstance(stmt, (ast.AugAssign, ast.AnnAssign)) and stmt.value is not None:
                kind = "augassign" if isinstance(stmt, ast.AugAssign) else "assign"
                self._store_target(stmt.target, kind, ref, stmt, stmt.value)
            elif isinstance(stmt, (ast.For, ast.AsyncFor)):
                for n in _names_in_target(stmt.target):
                    self.stores[n].append(_Store(n, "loop", ref, stmt))
            elif isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                self.stores[stmt.name].append(_Store(stmt.name, "function", ref, stmt))
            elif isinstance(stmt, (ast.Import, ast.ImportFrom)):
                for a in stmt.names:
                    if a.name == "*":
                        self.star_imports.append(stmt.module or "")
                        continue
                    n = a.asname or a.name.split(".")[0]
                    self.stores[n].append(_Store(n, "import", ref, stmt))

        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                if isinstance(node.ctx, ast.Load):
                    fn = _enclosing_function(node)
                    self.loads.append(_Load(node.id, self.ref(cell, node), node, fn))
                else:
                    self.all_bound.add(node.id)
            elif isinstance(node, ast.arg):
                self.all_bound.add(node.arg)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                self.all_bound.add(node.name)
            elif isinstance(node, ast.alias) and node.name != "*":
                self.all_bound.add(node.asname or node.name.split(".")[0])
            elif isinstance(node, ast.ExceptHandler) and node.name:
                self.all_bound.add(node.name)

    def _store_target(self, target: ast.AST, kind: str, ref: Ref, stmt: ast.AST, value: ast.AST) -> None:
        """Record an assignment target; x[...] = v and x.attr = v count as mutations of x."""
        if isinstance(target, ast.Name):
            self.stores[target.id].append(_Store(target.id, kind, ref, stmt, value))
        elif isinstance(target, (ast.Tuple, ast.List)):
            for n in _names_in_target(target):
                self.stores[n].append(_Store(n, kind, ref, stmt, None))
        else:  # x[...] = v, x.attr = v  -> mutation of x
            base = _base_name(target)
            if base:
                self.stores[base].append(_Store(base, "mutate", ref, stmt, value))

    def collect_functions(self, cell: Cell, tree: ast.Module) -> None:
        """For each module-level function: its parameters, the globals it reads or mutates,
        the arguments it mutates and the notebook functions it calls.
        """
        module_names = set(self.stores)
        for stmt in _module_statements(tree.body):
            if not isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            a = stmt.args
            params = [x.arg for x in a.posonlyargs + a.args + a.kwonlyargs]
            params += [x.arg for x in (a.vararg, a.kwarg) if x]
            local = set(params) | {
                n.id for n in ast.walk(stmt) if isinstance(n, ast.Name) and not isinstance(n.ctx, ast.Load)
            }
            fn = _Function(stmt.name, self.ref(cell, stmt), stmt, params, local)
            for n in ast.walk(stmt):
                if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load):
                    if n.id not in local and n.id in module_names and n.id != stmt.name:
                        kinds = {s.kind for s in self.stores[n.id]}
                        if "function" in kinds:
                            fn.calls.add(n.id)
                        elif "import" not in kinds:
                            fn.global_reads.add(n.id)
                elif isinstance(n, (ast.Subscript, ast.Attribute)) and not isinstance(n.ctx, ast.Load):
                    self._mutation(fn, _base_name(n), module_names)
                elif isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute):
                    if n.func.attr in MUTATING_METHODS:
                        self._mutation(fn, _base_name(n.func.value), module_names)
            self.functions[stmt.name] = fn

    def _mutation(self, fn: _Function, base: str | None, module_names: set[str]) -> None:
        """Classify a mutation inside fn as one of its arguments or a module-level global."""
        if not base:
            return
        if base in fn.params:
            fn.arg_mutations.add(base)
        elif base not in fn.locals and base in module_names:
            fn.global_mutations.add(base)

    # -- fact producers --

    def widgets(self) -> None:
        """Notebook parameters declared with dbutils.widgets.*, with default and choices."""
        for cell, tree in self.trees:
            for call in _calls(tree):
                dotted = _dotted(call.func)
                if not dotted or not dotted.startswith("dbutils.widgets."):
                    continue
                kind = dotted.rsplit(".", 1)[1]
                args = [_literal(a)[1] for a in call.args]
                if kind == "get":
                    continue
                name = args[0] if args else None
                default = args[1] if len(args) > 1 else None
                choices = args[2] if len(args) > 2 else None
                extra = f", choices {choices}" if choices else ""
                self.add(FactKind.WIDGET, self.ref(cell, call),
                         f"Notebook parameter (widget) {name!r}: {kind}, default {default!r}{extra}",
                         name=name, widget=kind, default=default, choices=choices)

    def constants(self) -> None:
        """Module-level literal assignments, and constants assigned more than once.

        A reassignment gets its own fact, and _effective_values says which value is in
        effect at each use: the trap where a later cell silently overrides a setting.
        """
        for name, stores in self.stores.items():
            module_assigns = [s for s in stores if s.kind in ("assign", "augassign", "mutate")]
            literal_assigns = [s for s in module_assigns if s.kind == "assign" and s.value is not None
                               and _literal(s.value)[0] and _is_meaningful_literal(name, s.value)]
            for s in literal_assigns:
                self.add(FactKind.CONSTANT, s.ref, f"{name} = {_short(s.value)}",
                         name=name, value=_literal(s.value)[1])
            plain = [s for s in module_assigns if s.kind in ("assign", "augassign")]
            if not literal_assigns or len(plain) < 2:
                continue
            first = plain[0]
            for later in plain[1:]:
                self.add(FactKind.CONSTANT_REASSIGNED, later.ref,
                         f"{name} is reassigned to {_short(later.value) if later.value else '?'} "
                         f"in cell {later.ref.cell}; it was first set to "
                         f"{_short(first.value) if first.value else '?'} at {first.ref}. "
                         f"Code after line {later.ref.start} sees the new value.",
                         name=name, first_ref=str(first.ref), first_line=first.ref.start)
            self._effective_values(name, plain)

    def _effective_values(self, name: str, assigns: list[_Store]) -> None:
        """For a reassigned constant, which assignment is in effect at each use."""
        lines = sorted(s.ref.start for s in assigns)

        def effective(at: int) -> _Store | None:
            """The latest assignment before line `at`, or None if there is none."""
            prior = [s for s in assigns if s.ref.start < at]
            return max(prior, key=lambda s: s.ref.start) if prior else None

        uses: list[tuple[Ref, str]] = []
        for ld in self.loads:
            if ld.name == name and ld.function is None and ld.ref.start not in lines:
                uses.append((ld.ref, f"{name} is read here"))
        readers = {f.name for f in self.functions.values() if name in f.global_reads}
        for fname in readers:
            for site in self._module_call_sites(fname):
                uses.append((site, f"{fname}() reads global {name} when called here"))
        for ref, what in sorted(uses):
            eff = effective(ref.start)
            if eff is None:
                summary = f"{what}, but it is not assigned before this line"
            else:
                summary = (f"{what}; value in effect is {_short(eff.value) if eff.value else '?'} "
                           f"(assigned at line {eff.ref.start})")
                if eff is not assigns[0]:
                    summary += f", not the original {_short(assigns[0].value)} from line {assigns[0].ref.start}"
            self.add(FactKind.EFFECTIVE_VALUE, ref, summary, name=name,
                     assigned_line=eff.ref.start if eff else None)

    def _module_call_sites(self, fname: str, seen: frozenset[str] = frozenset()) -> list[Ref]:
        """Lines at module level where fname runs, directly or via other functions."""
        sites = []
        for ld in self.loads:
            if ld.name != fname:
                continue
            if ld.function is None:
                sites.append(ld.ref)
            elif ld.function not in seen and ld.function != fname:
                sites += self._module_call_sites(ld.function, seen | {fname})
        return sites

    def reads_and_writes(self) -> None:
        """Spark file and table reads, table writes and a few side-effect calls
        (dbutils.notebook.exit, spark.conf.set) in Python cells.
        """
        for cell, tree in self.trees:
            for call in _calls(tree):
                if not isinstance(call.func, ast.Attribute):
                    continue
                method = call.func.attr
                receiver = ast.unparse(call.func.value)
                chain = _call_chain(call)
                ref = self.ref(cell, call)
                target = _assigned_names(call)
                into = f" into {', '.join(target)}" if target else ""

                if method in READ_METHODS and receiver.startswith("spark.read"):
                    src = ast.unparse(call.args[0]) if call.args else "?"
                    opts = {k: v for k, v in chain.get("option", [])}
                    opt_txt = f" with options {opts}" if opts else ""
                    kind = FactKind.TABLE_READ if method == "table" else FactKind.FILE_READ
                    self.add(kind, ref, f"Reads {method.upper() if method != 'table' else 'table'} "
                                        f"{src}{opt_txt}{into}", method=method, source=src,
                             options=opts, assigned_to=target)
                elif _dotted(call.func) == "spark.table":
                    src = ast.unparse(call.args[0]) if call.args else "?"
                    self.add(FactKind.TABLE_READ, ref, f"Reads via spark.table({src}){into}",
                             source=src, assigned_to=target)
                elif _dotted(call.func) == "spark.sql":
                    self._spark_sql(call, ref, target)
                elif method in WRITE_METHODS and ".write" in "." + receiver:
                    self._write(cell, call, method, chain, ref)

                if _dotted(call.func) in ("dbutils.notebook.exit", "spark.conf.set"):
                    self._side_effect(call, ref, _dotted(call.func))

    def _write(self, cell: Cell, call: ast.Call, method: str, chain: dict, ref: Ref) -> None:
        """A DataFrame write: target table (resolved through a string constant if possible),
        mode, format, options and the if/else conditions it runs under.
        """
        arg = call.args[0] if call.args else None
        table = ast.unparse(arg) if arg is not None else "?"
        via = ""
        if isinstance(arg, ast.Name):
            val = self.literal(arg.id, before_line=ref.start)
            if isinstance(val, str):
                via = f" (= {arg.id}, set elsewhere)"
                table = val
        elif isinstance(arg, ast.Constant):
            table = arg.value
        mode = _first_arg(chain.get("mode")) or "default (errorIfExists)"
        fmt = _first_arg(chain.get("format"))
        opts = dict(chain.get("option", []))
        guards = _guards(call)
        guard_txt = f"only when {' and '.join(guards)}" if guards else "unconditionally"
        self.add(FactKind.TABLE_WRITE, ref,
                 f"Writes {table!s}{via} with {method}, mode {mode}"
                 f"{', format ' + str(fmt) if fmt else ''}{', options ' + str(opts) if opts else ''}, "
                 f"{guard_txt}",
                 table=table, method=method, mode=mode, format=fmt, options=opts, guards=guards)

    def _spark_sql(self, call: ast.Call, ref: Ref, target: list[str]) -> None:
        """Classify a spark.sql(...) statement: table writes (with guards), table reads,
        or another statement such as USE or SET, which is reported as a side effect."""
        arg = call.args[0] if call.args else None
        src = ast.unparse(arg) if arg is not None else "?"
        into = f" into {', '.join(target)}" if target else ""
        guards = _guards(call)
        guard_txt = f"only when {' and '.join(guards)}" if guards else "unconditionally"
        text = self._sql_text(arg, ref.start)
        if text is None:
            self.add(FactKind.SIDE_EFFECT, ref,
                     f"Runs SQL via spark.sql({src}) {guard_txt}; the statement is not known statically",
                     call="spark.sql", guards=guards)
            return
        writes, reads = sql_tables(text)
        for op, name in writes:
            self.add(FactKind.TABLE_WRITE, ref, f"spark.sql runs {op} {name} {guard_txt}",
                     table=name, method="sql", mode=op, guards=guards)
        for name in reads:
            self.add(FactKind.TABLE_READ, ref, f"spark.sql reads table {name}{into}",
                     source=name, assigned_to=target)
        if writes or reads:
            return
        statement = " ".join(text.split())
        if statement.split(" ", 1)[0].upper() in _SQL_READ_ONLY:
            self.add(FactKind.TABLE_READ, ref, f"Reads via spark.sql: {statement[:100]}{into}",
                     source=src, assigned_to=target)
        else:
            self.add(FactKind.SIDE_EFFECT, ref, f"Runs SQL via spark.sql: {statement[:100]} {guard_txt}",
                     call="spark.sql", guards=guards)

    def _sql_text(self, arg: ast.AST | None, before_line: int) -> str | None:
        """The SQL text of a spark.sql argument; f-string fields become {expr}."""
        if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
            return arg.value
        if isinstance(arg, ast.JoinedStr):
            return "".join(v.value if isinstance(v, ast.Constant) else "{" + ast.unparse(v.value) + "}"
                           for v in arg.values)
        if isinstance(arg, ast.Name):
            value = self.literal(arg.id, before_line=before_line)
            return value if isinstance(value, str) else None
        return None

    def _side_effect(self, call: ast.Call, ref: Ref, what: str) -> None:
        guards = _guards(call)
        guard_txt = f"only when {' and '.join(guards)}" if guards else "unconditionally"
        self.add(FactKind.SIDE_EFFECT, ref, f"Calls {what}(...) {guard_txt}", call=what, guards=guards)

    def run_cells_and_sql(self) -> None:
        """%run dependencies, and table reads and writes in SQL cells, line by line."""
        for cell in self.nb.cells:
            if cell.kind is CellKind.RUN:
                target = (cell.magic or "").split(maxsplit=1)[1:] or ["?"]
                self.add(FactKind.RUN_DEPENDENCY, Ref(cell.number, cell.start_line, cell.end_line),
                         f"Runs notebook {target[0]} inline (%run); every name it defines "
                         f"becomes available here, but its code is not part of this file",
                         notebook=target[0])
            elif cell.kind is CellKind.SQL:
                for i, line in enumerate(cell.code_lines):
                    n = cell.start_line + i
                    stripped = line.strip()
                    if stripped.startswith("--"):
                        continue
                    writes, reads = sql_tables(stripped)
                    for op, name in writes:
                        self.add(FactKind.TABLE_WRITE, Ref(cell.number, n, n),
                                 f"SQL runs {op} {name}", table=name, method="sql", mode=op)
                    for name in reads:
                        self.add(FactKind.TABLE_READ, Ref(cell.number, n, n),
                                 f"SQL reads table {name}", source=name)

    def scala_cells(self) -> None:
        """Table reads and writes in Scala cells, found by regular expressions.

        Conditions around a statement are not analysed, so the facts say so
        instead of claiming the statement runs unconditionally."""
        for cell in self.nb.cells:
            if cell.kind is not CellKind.SCALA or cell.commented_out:
                continue
            # Blank "//" comment lines, keeping offsets so matches map to file lines.
            code = "\n".join(" " * len(c) if c.lstrip().startswith("//") else c for c in cell.code_lines)

            def ref(start: int, end: int) -> Ref:
                """Ref for a match between two offsets of the cell's joined code."""
                first = cell.start_line + code.count("\n", 0, start)
                return Ref(cell.number, first, cell.start_line + code.count("\n", 0, max(start, end - 1)))

            for m in _SCALA_SQL_RE.finditer(code):
                r = ref(m.start(), m.end())
                if not m["lit"]:
                    self.add(FactKind.SIDE_EFFECT, r, "Scala runs SQL via spark.sql(...); the statement "
                             "is not known statically and its conditions are not analysed",
                             call="spark.sql", guards=["not analysed (Scala)"])
                    continue
                text = _scala_string_value(m["lit"])
                writes, reads = sql_tables(text)
                for op, name in writes:
                    self.add(FactKind.TABLE_WRITE, r, f"Scala spark.sql runs {op} {name} "
                             f"(conditions not analysed)", table=name, method="sql", mode=op,
                             guards=["not analysed (Scala)"])
                for name in reads:
                    self.add(FactKind.TABLE_READ, r, f"Scala spark.sql reads table {name}", source=name)
                if not writes and not reads:
                    statement = " ".join(text.split())
                    kind = (FactKind.TABLE_READ if statement.split(" ", 1)[0].upper() in _SQL_READ_ONLY
                            else FactKind.SIDE_EFFECT)
                    self.add(kind, r, f"Scala runs SQL via spark.sql: {statement[:100]}",
                             source=statement[:100], call="spark.sql", guards=["not analysed (Scala)"])
            for m in _SCALA_WRITE_RE.finditer(code):
                self.add(FactKind.TABLE_WRITE, ref(m.start(), m.end()),
                         f"Scala writes table {m['name']} with {m['method']} (conditions not analysed)",
                         table=m["name"], method=m["method"], guards=["not analysed (Scala)"])
            for m in _SCALA_READ_RE.finditer(code):
                self.add(FactKind.TABLE_READ, ref(m.start(), m.end()),
                         f"Scala reads table {m['name']}", source=m["name"])

    def external_names(self) -> None:
        """Names used but never defined in the notebook: Databricks globals, names that a
        previous %run presumably provides, or truly undefined names. Calls to the latter
        two are also reported as side effects, since their code is not visible.
        """
        run_cells = [c for c in self.nb.cells if c.kind is CellKind.RUN]
        first_use: dict[str, _Load] = {}
        for ld in self.loads:
            if ld.name in self.all_bound or ld.name in BUILTINS:
                continue
            if ld.name not in first_use or ld.ref < first_use[ld.name].ref:
                first_use[ld.name] = ld
        for name, ld in sorted(first_use.items(), key=lambda kv: kv[1].ref):
            uses = sorted({l.ref.start for l in self.loads if l.name == name})
            star = next((m for m in self.star_imports if STAR_EXPORTS.get(m, lambda n: False)(name)), None)
            if name in DATABRICKS_GLOBALS:
                origin, src = "Databricks runtime global", "databricks"
            elif star:
                origin, src = f"from star import of {star}", "star_import"
            else:
                prior = [c for c in run_cells if c.end_line < ld.ref.start]
                if prior:
                    nbk = (prior[-1].magic or "").split(maxsplit=1)[1:] or ["?"]
                    origin, src = f"not defined in this notebook; presumably provided by %run {nbk[0]}", "run"
                else:
                    origin, src = "not defined anywhere in this notebook", "undefined"
            if src == "star_import":
                continue  # ordinary library names, not interesting as facts
            self.add(FactKind.EXTERNAL_NAME, ld.ref,
                     f"{name} is {origin} (used on lines {_ranges(uses)})",
                     name=name, origin=src, lines=uses)
            if src in ("run", "undefined"):
                for l in self.loads:
                    if l.name == name and isinstance(getattr(l.node, "_cp_parent", None), ast.Call) \
                            and l.node._cp_parent.func is l.node:
                        self._side_effect(l.node._cp_parent, _call_ref(self, l),
                                          f"external function {name}")

    def function_facts(self) -> None:
        """One fact per function summarising its signature, globals, calls and call sites."""
        for fn in self.functions.values():
            calls = sorted({l.ref.start for l in self.loads if l.name == fn.name})
            parts = [f"def {fn.name}({', '.join(fn.params)})"]
            if fn.global_reads:
                parts.append(f"reads globals {sorted(fn.global_reads)}")
            if fn.global_mutations:
                parts.append(f"MUTATES globals {sorted(fn.global_mutations)}")
            if fn.arg_mutations:
                parts.append(f"mutates arguments {sorted(fn.arg_mutations)}")
            if fn.calls:
                parts.append(f"calls {sorted(fn.calls)}")
            parts.append(f"used on lines {_ranges(calls)}" if calls else "never called")
            self.add(FactKind.FUNCTION, fn.ref, "; ".join(parts), name=fn.name, params=fn.params,
                     global_reads=sorted(fn.global_reads), global_mutations=sorted(fn.global_mutations),
                     arg_mutations=sorted(fn.arg_mutations), calls=sorted(fn.calls), used_on=calls)

    def unused(self) -> None:
        """Imports, variables and functions that are defined but never read. Names starting
        with an underscore are skipped as intentionally private.
        """
        loaded = {l.name for l in self.loads}
        for name, stores in self.stores.items():
            if name in loaded or name.startswith("_"):
                continue
            kinds = {s.kind for s in stores}
            first = stores[0]
            if kinds == {"import"}:
                self.add(FactKind.UNUSED_IMPORT, first.ref, f"Import {name} is never used", name=name)
            elif kinds & {"assign", "augassign", "function"} and "loop" not in kinds:
                what = "Function" if "function" in kinds else "Variable"
                self.add(FactKind.UNUSED_NAME, first.ref,
                         f"{what} {name} is defined but never used", name=name)

    def commented_code(self) -> None:
        """Commented-out code: whole cells, blocks of Python comment lines that parse as
        code, and commented SQL statements. Plain prose comments are not reported.
        """
        for cell in self.nb.cells:
            whole = Ref(cell.number, cell.start_line, cell.end_line)
            if cell.kind is CellKind.PYTHON and cell.commented_out:
                code_lines = [c for c in cell.code_lines if _is_code_comment(c)]
                if code_lines:
                    self.add(FactKind.COMMENTED_CODE, whole,
                             f"Entire cell is commented-out code ({len(code_lines)} code lines), "
                             f"starting: {cell.code_lines[0].strip()[:80]}", whole_cell=True)
                continue
            if cell.kind is CellKind.PYTHON:
                block: list[int] = []
                for i, c in enumerate(cell.code_lines + ("",)):
                    if _is_code_comment(c):
                        block.append(cell.start_line + i)
                    elif block:
                        text = cell.code_line(block[0]).strip()
                        self.add(FactKind.COMMENTED_CODE, Ref(cell.number, block[0], block[-1]),
                                 f"Commented-out code: {text[:100]}", whole_cell=False)
                        block = []
            elif cell.kind is CellKind.SQL:
                for i, c in enumerate(cell.code_lines):
                    s = c.strip()
                    if s.startswith("--") and SQL_CODE_RE.match(s[2:]):
                        n = cell.start_line + i
                        self.add(FactKind.COMMENTED_CODE, Ref(cell.number, n, n),
                                 f"Commented-out SQL: {s[:100]}", whole_cell=False)

    def thresholds(self) -> None:
        """Numeric literals used in comparisons: the business tolerances hidden in code."""
        for cell, tree in self.trees:
            for node in ast.walk(tree):
                if not isinstance(node, ast.Compare):
                    continue
                operands = [node.left, *node.comparators]
                nums = [v for ok, v in map(_literal, operands)
                        if ok and isinstance(v, (int, float)) and not isinstance(v, bool)
                        and v not in (0, 1, -1)]
                if not nums:
                    continue
                fn = _enclosing_function(node)
                where = f" (in {fn})" if fn else ""
                self.add(FactKind.THRESHOLD, self.ref(cell, node),
                         f"Hard-coded threshold: {ast.unparse(node)}{where}",
                         values=nums, expression=ast.unparse(node), function=fn)

    def dataflow(self) -> None:
        """For each module-level assignment, the notebook variables it is built or updated
        from. Imports and loop variables are ignored as noise.
        """
        skip = {n for n, ss in self.stores.items()
                if any(s.kind == "import" for s in ss) or {s.kind for s in ss} == {"loop"}}
        external = {f.data["name"] for f in self.facts
                    if f.kind is FactKind.EXTERNAL_NAME and f.data["origin"] in ("run", "undefined")}
        module_names = (set(self.stores) - skip) | external
        for name, stores in self.stores.items():
            for s in stores:
                if s.kind not in ("assign", "augassign", "mutate") or s.value is None:
                    continue
                comp_vars = {n for c in ast.walk(s.stmt) if isinstance(c, ast.comprehension)
                             for n in _names_in_target(c.target)}
                comp_vars |= {a.arg for lam in ast.walk(s.stmt) if isinstance(lam, ast.Lambda)
                              for a in lam.args.args}
                uses = sorted({n.id for n in ast.walk(s.stmt) if isinstance(n, ast.Name)
                               and isinstance(n.ctx, ast.Load) and n.id in module_names}
                              - {name} - comp_vars)
                if not uses:
                    continue
                updated = s.kind in ("mutate", "augassign") or name in _loads_in(s.value)
                verb = "is updated from" if updated else "is built from"
                self.add(FactKind.DATAFLOW, s.ref, f"{name} {verb} {', '.join(uses)}",
                         target=name, uses=uses, mutation=s.kind == "mutate")


# --- AST utilities ----------------------------------------------------------


def analyze(nb: Notebook) -> FactSheet:
    """Static fact sheet for a parsed notebook. Never raises on bad code: cells that do
    not parse become parse_error facts.
    """
    return _Analyzer(nb).run()


def _annotate_parents(tree: ast.AST) -> None:
    """Store each node's parent as `_cp_parent`, which the ast module does not provide."""
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            child._cp_parent = node  # type: ignore[attr-defined]


def _parents(node: ast.AST) -> Iterator[tuple[ast.AST, ast.AST]]:
    """Yield (parent, child) pairs walking up from node."""
    child = node
    parent = getattr(node, "_cp_parent", None)
    while parent is not None:
        yield parent, child
        child, parent = parent, getattr(parent, "_cp_parent", None)


def _enclosing_function(node: ast.AST) -> str | None:
    """Name of the outermost function containing node, or None at module level."""
    name = None
    for parent, _ in _parents(node):
        if isinstance(parent, (ast.FunctionDef, ast.AsyncFunctionDef)):
            name = parent.name  # keep climbing: we want the outermost (module-level) one
    return name


def _guards(node: ast.AST) -> list[str]:
    """Conditions a node runs under, outermost first. Being in an else branch gives
    "not (test)". Used to tell guarded writes from unconditional ones.
    """
    out = []
    for parent, child in _parents(node):
        if isinstance(parent, (ast.If, ast.While)):
            test = ast.unparse(parent.test)
            if any(child is s for s in parent.body):
                out.append(test)
            elif any(child is s for s in parent.orelse):
                out.append(f"not ({test})")
        elif isinstance(parent, ast.IfExp):
            test = ast.unparse(parent.test)
            out.append(test if child is parent.body else f"not ({test})" if child is parent.orelse else test)
    return list(reversed(out))


def _module_statements(body: list[ast.stmt]) -> Iterator[ast.stmt]:
    """Statements at module scope, including inside if/for/while/with/try, not functions."""
    for stmt in body:
        yield stmt
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
            continue
        for attr in ("body", "orelse", "finalbody"):
            yield from _module_statements(getattr(stmt, attr, []) or [])
        for h in getattr(stmt, "handlers", []) or []:
            yield from _module_statements(h.body)


def _names_in_target(t: ast.AST) -> list[str]:
    """All names in an assignment target, including tuple unpacking."""
    return [n.id for n in ast.walk(t) if isinstance(n, ast.Name)]


def _base_name(node: ast.AST | None) -> str | None:
    """The variable at the root of x.a[0].b(...), or None if it is not a plain name."""
    while isinstance(node, (ast.Subscript, ast.Attribute, ast.Call)):
        node = node.func if isinstance(node, ast.Call) else node.value
    return node.id if isinstance(node, ast.Name) else None


def _loads_in(node: ast.AST) -> set[str]:
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}


def _calls(tree: ast.AST) -> Iterator[ast.Call]:
    return (n for n in ast.walk(tree) if isinstance(n, ast.Call))


def _dotted(node: ast.AST) -> str | None:
    """Dotted name of an attribute chain ("spark.read.csv"), or None if it does not start with a name."""
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return None


def _call_chain(call: ast.Call) -> dict[str, list]:
    """Method calls in a fluent chain below this call: {"mode": [("overwrite",)], "option": [(k, v)]}."""
    chain: dict[str, list] = defaultdict(list)
    node: ast.AST = call.func.value if isinstance(call.func, ast.Attribute) else call.func
    while True:
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            args = tuple(_literal(a)[1] if _literal(a)[0] else ast.unparse(a) for a in node.args)
            chain[node.func.attr].append(args)
            node = node.func.value
        elif isinstance(node, ast.Attribute):
            node = node.value
        else:
            break
    return chain


def _first_arg(calls: list | None) -> Any:
    """First argument of the first call in a _call_chain entry, or None."""
    return calls[0][0] if calls and calls[0] else None


def _assigned_names(node: ast.AST) -> list[str]:
    """Names that the statement containing node assigns to, for "read ... into df"."""
    for parent, _ in _parents(node):
        if isinstance(parent, ast.Assign):
            return [n for t in parent.targets for n in _names_in_target(t)]
        if isinstance(parent, ast.stmt):
            return []
    return []


def _literal(node: ast.AST) -> tuple[bool, Any]:
    """(True, value) if node is a Python literal, else (False, None). Never raises."""
    try:
        return True, ast.literal_eval(node)
    except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
        return False, None


def _is_meaningful_literal(name: str, value: ast.AST) -> bool:
    """Skip empty initialisers like lines = [] or h = "" unless the name looks like a constant."""
    if name.isupper():
        return True
    _, v = _literal(value)
    return not (isinstance(v, (list, dict, tuple, set, str)) and len(v) == 0)


def _short(node: ast.AST | None, limit: int = 80) -> str:
    """Source text of node, shortened to `limit` characters for fact summaries."""
    if node is None:
        return "?"
    s = ast.unparse(node)
    return s if len(s) <= limit else s[: limit - 3] + "..."


def _ranges(lines: list[int]) -> str:
    """Comma-separated line numbers, or "none"."""
    return ", ".join(str(n) for n in lines) if lines else "none"


def _call_ref(an: _Analyzer, load: _Load) -> Ref:
    """Ref of the call in which a loaded name is the function being called."""
    call = load.node._cp_parent  # type: ignore[attr-defined]
    cell = an.nb.cell(load.ref.cell)
    return an.ref(cell, call)


def _scala_string_value(literal: str) -> str:
    """Text of a Scala string literal; interpolations ($x, ${x}) become {x}."""
    interpolated = literal[0] in "sf"
    body = literal[1:] if interpolated else literal
    body = body[3:-3] if body.startswith('"""') else body[1:-1]
    if interpolated:
        body = _SCALA_INTERPOLATION_RE.sub(lambda m: "{" + (m[1] or m[2]) + "}", body)
    return body


def sql_tables(sql: str) -> tuple[list[tuple[str, str]], list[str]]:
    """Tables a SQL text modifies, as (operation, name), and tables it reads.

    Regex-based: good for table names, blind to CTE names and SQL structure.
    A table a statement writes is not also reported as read (DELETE FROM t).
    """
    writes = [(" ".join(m["op"].split()).upper(), m["name"]) for m in SQL_WRITE_RE.finditer(sql)]
    written = {name for _, name in writes}
    reads = [m["name"] for m in SQL_TABLE_RE.finditer(sql) if m["name"] not in written]
    return writes, reads


def _is_code_comment(line: str) -> bool:
    """A comment line whose text parses as a Python statement that looks like code."""
    s = line.strip()
    if not s.startswith("#"):
        return False
    body = s.lstrip("#").strip()
    if not body or not any(ch in body for ch in "=()[]"):
        return False
    try:
        ast.parse(body)
        return True
    except SyntaxError:
        # Block openers ("for x in y:", "def f():", "if a:") don't parse on their own.
        try:
            ast.parse(body + "\n    pass")
            return True
        except SyntaxError:
            return False
