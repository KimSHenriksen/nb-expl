"""Claims: citable statements about a notebook, from static analysis and from the LLM.

Flow:
1. Every analysis fact becomes a claim (source "analysis"); these are correct by
   construction.
2. The notebook is split into chunks of consecutive cells; the LLM extracts
   claims per chunk, plus one pass over the whole notebook for cross-cell claims.
3. Every LLM citation is checked mechanically: the ref must lie inside its cell
   and the quote must occur verbatim (modulo whitespace) in the cited lines.
   Small, unambiguous mistakes are repaired (wrong cell number, lines a little
   off, quote found in exactly one place) and the repair is recorded.
4. Claims with no valid citation are sent back once for correction; whatever
   still fails is kept in `rejected` with the reason, never shown as fact.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from .analysis import Fact, FactKind, FactSheet, analyze
from .config import Settings
from .llm import LLMClient
from .parser import Cell, CellKind, Notebook
from .utils import Progress, chat_messages, fill, load_prompt, parallel_map, reporter
from .refs import Ref, check_ref, parse_ref

log = logging.getLogger(__name__)

Topic = Literal[
    "purpose", "input", "parameter", "output", "business_rule", "calculation", "filter",
    "data_flow", "data_quality", "structure", "dependency", "side_effect", "dead_code",
    "risk", "invariant",
]
Importance = Literal["high", "medium", "low"]
Certainty = Literal["explicit", "inferred"]

# --- what the LLM returns ---------------------------------------------------


class LLMCitation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    ref: str = Field(description="C<cell>:L<start>-<end> using file line numbers")
    quote: str = Field(description="Exact text copied from the cited lines")


class LLMClaim(BaseModel):
    model_config = ConfigDict(extra="forbid")
    topic: Topic
    statement: str
    importance: Importance
    certainty: Certainty
    citations: list[LLMCitation]


class LLMClaimList(BaseModel):
    model_config = ConfigDict(extra="forbid")
    claims: list[LLMClaim]


# --- validated claims ---------------------------------------------------------


class Citation(BaseModel):
    ref: str
    quote: str
    note: str | None = None
    """Set when the citation was repaired, e.g. "cell number corrected"."""


class LLMVerdict(BaseModel):
    """Field order matters: the model describes the code before it decides."""
    model_config = ConfigDict(extra="forbid")
    id: str
    code_says: str
    reason: str
    verdict: Literal["supported", "partially_supported", "unsupported"]
    corrected_statement: str


class LLMVerdictList(BaseModel):
    model_config = ConfigDict(extra="forbid")
    verdicts: list[LLMVerdict]


Verification = Literal["static", "verified", "corrected", "unverified"]


class Claim(BaseModel):
    id: str
    source: Literal["analysis", "llm"]
    topic: Topic
    statement: str
    importance: Importance
    certainty: Certainty
    citations: list[Citation]
    verification: Verification = "unverified"
    """static: computed from the AST; verified/corrected: checked by a second LLM
    pass against the cited code; unverified: not (successfully) checked."""
    verification_note: str | None = None
    original_statement: str | None = None
    """The statement before verification corrected it."""


class RejectedClaim(BaseModel):
    statement: str
    topic: str
    errors: list[str]
    stage: Literal["citation", "verification"] = "citation"


class ClaimSet(BaseModel):
    claims: list[Claim]
    rejected: list[RejectedClaim]
    stats: dict[str, int]
    errors: list[str] = []
    """Extraction jobs that failed entirely (e.g. the LLM returned no valid JSON)."""

    def by_id(self) -> dict[str, Claim]:
        return {c.id: c for c in self.claims}


# --- citation validation ------------------------------------------------------

_LINE_PREFIX_RE = re.compile(r"^\s*\d+\s*\|\s?", re.MULTILINE)


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def clean_quote(quote: str) -> str:
    """Normalise a model's quote before matching it against the code.

    Undoes common copying mistakes: surrounding quotes, double-escaped newlines and quotes,
    the "123| " line gutter from the prompt and leading or trailing "...".
    """
    q = quote.strip()
    if len(q) >= 2 and q[0] == q[-1] and q[0] in "`'\"":
        q = q[1:-1]
    # Models sometimes double-escape inside JSON strings: a literal \n or \" in the text.
    q = q.replace("\\n", "\n").replace('\\"', '"').replace("\\'", "'")
    q = _LINE_PREFIX_RE.sub("", q)  # model copied the "123| " gutter
    q = re.sub(r"^\s*\.\.\.|\.\.\.\s*$", "", q)
    return _norm(q)


def _range_text(nb: Notebook, start: int, end: int) -> list[str]:
    """Normalised raw and MAGIC-stripped text of a line range, for quote matching."""
    raw = " ".join(nb.line(n) for n in range(start, end + 1))
    cell = nb.cell_for_line(start)
    variants = [_norm(raw)]
    if cell and cell.contains(end):
        variants.append(_norm(" ".join(cell.code_line(n) for n in range(start, end + 1))))
    return variants


def quote_in_range(nb: Notebook, ref: Ref, quote: str) -> bool:
    """True if the cleaned quote occurs in the cited lines, with or without # MAGIC prefixes."""
    q = clean_quote(quote)
    return bool(q) and any(q in t for t in _range_text(nb, ref.start, ref.end))


def find_quote(nb: Notebook, quote: str, max_span: int = 12) -> list[Ref]:
    """Smallest line windows (within one cell) whose text contains the quote."""
    q = clean_quote(quote)
    if not q:
        return []
    hits: list[Ref] = []
    for cell in nb.cells:
        if cell.kind is CellKind.EMPTY:
            continue
        for start in range(cell.start_line, cell.end_line + 1):
            for end in range(start, min(cell.end_line, start + max_span - 1) + 1):
                if any(q in t for t in _range_text(nb, start, end)):
                    hits.append(Ref(cell.number, start, end))
                    break
    # keep minimal windows only (drop windows that contain another hit)
    return [h for h in hits if not any(o != h and h.start <= o.start and o.end <= h.end for o in hits)]


def validate_citation(nb: Notebook, ref_text: str, quote: str) -> tuple[Citation | None, str | None]:
    """Return (citation, None) if valid or repairable, else (None, error)."""
    q = clean_quote(quote)
    if len(q) < 3:
        return None, f"{ref_text}: quote is empty or too short"
    try:
        ref: Ref | None = parse_ref(ref_text)
    except ValueError:
        ref = None
    note = None

    if ref is not None and check_ref(ref, nb) is not None:
        owner = nb.cell_for_line(ref.start)
        if owner and owner.contains(ref.end) and ref.start <= ref.end:
            ref, note = Ref(owner.number, ref.start, ref.end), "cell number corrected"
        else:
            ref_err = check_ref(ref, nb)
            ref = None
            note = ref_err

    if ref is not None and quote_in_range(nb, ref, quote):
        return Citation(ref=str(ref), quote=q, note=note), None

    hits = find_quote(nb, quote)
    if ref is not None:
        near = [h for h in hits if h.cell == ref.cell and abs(h.start - ref.start) <= 10]
        if near:
            best = min(near, key=lambda h: abs(h.start - ref.start))
            return Citation(ref=str(best), quote=q, note=f"lines adjusted from {ref}"), None
    if len(hits) == 1:
        return Citation(ref=str(hits[0]), quote=q, note=f"relocated from {ref_text}"), None
    if hits:
        return None, (f"{ref_text}: quote {q[:60]!r} not in cited lines and appears in "
                      f"{len(hits)} other places")
    return None, f"{ref_text}: quote {q[:60]!r} does not appear in the notebook"


def validate_claims(nb: Notebook, raw: list[LLMClaim]) -> tuple[list[tuple[LLMClaim, list[Citation]]], list[tuple[LLMClaim, list[str]]]]:
    """Check every citation of each model claim.

    A claim is kept if at least one citation is valid or repairable; claims with none
    are returned with their errors so they can be sent back for repair.
    """
    ok, bad = [], []
    for c in raw:
        cites, errors = [], []
        for cit in c.citations:
            v, err = validate_citation(nb, cit.ref, cit.quote)
            if v and all(v.ref != x.ref for x in cites):
                cites.append(v)
            elif err:
                errors.append(err)
        if not c.citations:
            errors.append("no citations")
        if cites:
            ok.append((c, cites))
        else:
            bad.append((c, errors))
    return ok, bad


# --- analysis facts as claims -----------------------------------------------

_FACT_TOPIC: dict[FactKind, Topic] = {
    FactKind.WIDGET: "parameter",
    FactKind.CONSTANT: "business_rule",
    FactKind.CONSTANT_REASSIGNED: "risk",
    FactKind.EFFECTIVE_VALUE: "risk",
    FactKind.FILE_READ: "input",
    FactKind.TABLE_READ: "input",
    FactKind.TABLE_WRITE: "output",
    FactKind.SIDE_EFFECT: "side_effect",
    FactKind.RUN_DEPENDENCY: "dependency",
    FactKind.EXTERNAL_NAME: "dependency",
    FactKind.FUNCTION: "structure",
    FactKind.UNUSED_NAME: "dead_code",
    FactKind.UNUSED_IMPORT: "dead_code",
    FactKind.COMMENTED_CODE: "dead_code",
    FactKind.THRESHOLD: "business_rule",
    FactKind.DATAFLOW: "data_flow",
    FactKind.PARSE_ERROR: "risk",
}
_HIGH = {FactKind.CONSTANT_REASSIGNED, FactKind.EFFECTIVE_VALUE, FactKind.TABLE_WRITE, FactKind.PARSE_ERROR}
_LOW = {FactKind.DATAFLOW, FactKind.UNUSED_IMPORT, FactKind.EXTERNAL_NAME}


def fact_to_claim(fact: Fact, nb: Notebook, n: int) -> Claim:
    """Turn a static-analysis fact into a claim marked "static".

    Importance: reassignments, effective values, table writes and parse errors are
    high; so are side effects without a guard. Dataflow and external names are low.
    """
    importance: Importance = "high" if fact.kind in _HIGH else "low" if fact.kind in _LOW else "medium"
    if fact.kind is FactKind.SIDE_EFFECT and not fact.data.get("guards"):
        importance = "high"
    quote = next((nb.line(i).strip() for i in range(fact.ref.start, fact.ref.end + 1)
                  if nb.line(i).strip()), "")
    return Claim(
        id=f"A{n:03d}", source="analysis", topic=_FACT_TOPIC[fact.kind], statement=fact.summary,
        importance=importance, certainty="explicit", verification="static",
        citations=[Citation(ref=str(fact.ref), quote=_norm(quote))],
    )


# --- prompting ------------------------------------------------------------------


def render_cells(cells: list[Cell]) -> str:
    """Cells as numbered source lines, the form in which the model sees the code."""
    out = []
    for c in cells:
        if c.kind is CellKind.EMPTY:
            continue
        extra = " (entirely commented out)" if c.commented_out else ""
        title = f' "{c.title}"' if c.title else ""
        out.append(f"### Cell C{c.number}{title}: {c.kind}{extra}, lines {c.start_line}-{c.end_line}")
        width = len(str(c.end_line))
        out.extend(f"{c.start_line + i:>{width}}| {line}" for i, line in enumerate(c.raw_lines))
        out.append("")
    return "\n".join(out)


def outline(nb: Notebook) -> str:
    """One line per cell (number, lines, kind, first line), so a chunk sees the whole notebook."""
    rows = []
    for c in nb.cells:
        if c.kind is CellKind.EMPTY:
            continue
        first = next((l.strip() for l in c.code_lines if l.strip()), "")
        tag = "commented out" if c.commented_out else str(c.kind)
        rows.append(f"C{c.number} L{c.start_line}-{c.end_line} [{tag}] {first[:70]}")
    return "\n".join(rows)


def render_claim_for_review(nb: Notebook, claim: Claim, context: int) -> str:
    """Claim text plus its cited lines (marked '>') with context from the same cell.

    The whole cell is shown when it is short; otherwise `context` lines either side.
    """
    out = [f"## {claim.id}", f"Claim: {claim.statement}"]
    for cit in claim.citations:
        ref = parse_ref(cit.ref)
        cell = nb.cell(ref.cell)
        if cell.end_line - cell.start_line < 40:
            lo, hi = cell.start_line, cell.end_line
        else:
            lo = max(cell.start_line, ref.start - context)
            hi = min(cell.end_line, ref.end + context)
        width = len(str(hi))
        out.append(f"Cited {cit.ref} (cell C{cell.number}, {cell.kind}):")
        for n in range(lo, hi + 1):
            mark = ">" if ref.start <= n <= ref.end else " "
            out.append(f"{mark}{n:>{width}}| {nb.line(n)}")
    return "\n".join(out)


def chunk_cells(nb: Notebook, max_lines: int) -> list[list[Cell]]:
    """Group consecutive non-empty cells into chunks of about max_lines lines. Cells are
    never split, so a cell longer than max_lines becomes a chunk of its own.
    """
    chunks: list[list[Cell]] = [[]]
    size = 0
    for c in nb.cells:
        if c.kind is CellKind.EMPTY:
            continue
        n = c.end_line - c.start_line + 1
        if chunks[-1] and size + n > max_lines:
            chunks.append([])
            size = 0
        chunks[-1].append(c)
        size += n
    return [c for c in chunks if c]


_DEFINITION_KINDS = {FactKind.WIDGET, FactKind.CONSTANT, FactKind.CONSTANT_REASSIGNED, FactKind.EXTERNAL_NAME,
                     FactKind.RUN_DEPENDENCY, FactKind.FUNCTION}


def _facts_for(sheet: FactSheet, cells: set[int]) -> str:
    """Facts about the chunk's own cells, plus notebook-wide definitions (constants, widgets,
    functions, external names) so a chunk never concludes that something is undefined."""
    sub = FactSheet([f for f in sheet.facts if f.kind is not FactKind.DATAFLOW
                     and (f.ref.cell in cells or f.kind in _DEFINITION_KINDS)])
    return sub.render() if sub.facts else "(none)"


# --- extraction -------------------------------------------------------------


class ClaimExtractor:
    def __init__(self, settings: Settings, llm: LLMClient):
        self.settings = settings
        self.llm = llm
        self.system = load_prompt("extract_system")

    def run(self, nb: Notebook, sheet: FactSheet | None = None, progress: Progress | None = None) -> ClaimSet:
        """Static facts as claims, then model claims per chunk plus one whole-notebook pass,
        then verification.

        A failed chunk or batch is recorded in ClaimSet.errors instead of failing the run.
        Citations are validated and, if needed, repaired once before a claim is rejected.
        """
        say = reporter(progress, log)
        sheet = sheet or analyze(nb)
        claims = [fact_to_claim(f, nb, i) for i, f in enumerate(sheet.facts, start=1)]
        say(f"{len(claims)} facts from static analysis")

        jobs = []
        for cells in chunk_cells(nb, self.settings.chunk_lines):
            nums = {c.number for c in cells}
            label = f"C{min(nums)}-C{max(nums)}"
            lines = sum(c.end_line - c.start_line + 1 for c in cells)
            prompt = fill(load_prompt("extract_chunk"), outline=outline(nb), facts=_facts_for(sheet, nums),
                          cells=label, code=render_cells(cells), target=max(5, lines // 6))
            jobs.append((label, prompt))
        jobs.append(("whole notebook", fill(load_prompt("extract_global"), facts=sheet.render(),
                                            code=render_cells(list(nb.cells)), target=25)))

        errors: list[str] = []

        def work(job: tuple[str, str]) -> tuple[str, list[tuple[LLMClaim, list[Citation]]], list[tuple[LLMClaim, list[str]]]]:
            """Extract and validate claims for one chunk; returns (label, valid, rejected)."""
            label, prompt = job
            say(f"extracting claims: {label}")
            messages = chat_messages(self.system, prompt)
            try:
                raw = self.llm.structured(messages, LLMClaimList, self.settings.model_extract, "claims").claims
            except Exception as e:  # one failed chunk must not lose the others
                log.exception("claim extraction failed for %s", label)
                say(f"{label}: FAILED ({e})")
                errors.append(f"{label}: {e}")
                return label, [], []
            ok, bad = validate_claims(nb, raw)
            if bad:
                say(f"{label}: {len(ok)} valid, {len(bad)} sent back for citation repair")
                ok2, bad = self._repair(nb, messages, raw, bad)
                ok += ok2
            say(f"{label}: {len(ok)} valid claims, {len(bad)} rejected")
            return label, ok, bad

        results = parallel_map(work, jobs, self.settings.concurrency)

        rejected: list[RejectedClaim] = []
        n = 0
        for _, ok, bad in results:
            for c, cites in ok:
                n += 1
                claims.append(Claim(id=f"M{n:03d}", source="llm", topic=c.topic, statement=c.statement,
                                    importance=c.importance, certainty=c.certainty, citations=cites))
            rejected += [RejectedClaim(statement=c.statement, topic=c.topic, errors=e) for c, e in bad]

        if self.settings.verify_claims:
            claims, unsupported = self._verify(nb, sheet, claims, errors, say)
            rejected += unsupported

        llm_claims = [c for c in claims if c.source == "llm"]
        stats = {
            "analysis_claims": len(claims) - len(llm_claims),
            "llm_claims": len(llm_claims),
            "rejected": len(rejected),
            "rejected_by_verification": sum(1 for r in rejected if r.stage == "verification"),
            "verified": sum(1 for c in llm_claims if c.verification == "verified"),
            "corrected": sum(1 for c in llm_claims if c.verification == "corrected"),
            "unverified": sum(1 for c in llm_claims if c.verification == "unverified"),
            "repaired_citations": sum(1 for c in llm_claims for x in c.citations if x.note),
            "llm_calls": self.llm.calls,
            "cache_hits": self.llm.cache_hits,
            "failed_jobs": len(errors),
        }
        say(f"done: {stats}")
        return ClaimSet(claims=claims, rejected=rejected, stats=stats, errors=errors)

    def _verify(self, nb: Notebook, sheet: FactSheet, claims: list[Claim], errors: list[str],
                say: Progress) -> tuple[list[Claim], list[RejectedClaim]]:
        """Second pass: does the cited code support each LLM claim?"""
        todo = [c for c in claims if c.source == "llm"]
        size = max(1, self.settings.verify_batch_size)
        batches = [todo[i:i + size] for i in range(0, len(todo), size)]
        system = load_prompt("verify_system")
        facts = FactSheet([f for f in sheet.facts if f.kind is not FactKind.DATAFLOW]).render()

        def work(batch: list[Claim]) -> dict[str, LLMVerdict]:
            """Verify one batch of claims; returns verdicts by claim id, or {} if the call failed."""
            label = f"{batch[0].id}-{batch[-1].id}"
            say(f"verifying claims {label}")
            listing = "\n\n".join(render_claim_for_review(nb, c, self.settings.verify_context_lines)
                                  for c in batch)
            prompt = fill(load_prompt("verify_claims"), facts=facts, claims=listing,
                          ids=", ".join(c.id for c in batch))
            messages = chat_messages(system, prompt)
            try:
                out = self.llm.structured(messages, LLMVerdictList,
                                         self.settings.model_verify or self.settings.model_extract, "verdicts")
            except Exception as e:  # claims in a failed batch stay "unverified"
                log.exception("verification failed for %s", label)
                say(f"verification {label}: FAILED ({e})")
                errors.append(f"verification {label}: {e}")
                return {}
            return {v.id: v for v in out.verdicts}

        verdicts: dict[str, LLMVerdict] = {}
        for result in parallel_map(work, batches, self.settings.concurrency):
            verdicts.update(result)

        kept, rejected = [], []
        for c in claims:
            v = verdicts.get(c.id) if c.source == "llm" else None
            if v is None:
                kept.append(c)
            elif v.verdict == "supported":
                kept.append(c.model_copy(update={"verification": "verified", "verification_note": v.reason}))
            elif v.verdict == "partially_supported" and v.corrected_statement.strip():
                kept.append(c.model_copy(update={
                    "verification": "corrected", "verification_note": v.reason,
                    "statement": v.corrected_statement.strip(), "original_statement": c.statement,
                }))
            elif v.verdict == "partially_supported":  # no correction offered: keep, flagged
                kept.append(c.model_copy(update={"verification": "unverified",
                                                 "verification_note": f"partially supported: {v.reason}"}))
            else:
                rejected.append(RejectedClaim(statement=c.statement, topic=c.topic,
                                              errors=[f"verification: {v.reason}"], stage="verification"))
        say(f"verification: {sum(1 for c in kept if c.verification == 'verified')} supported, "
            f"{sum(1 for c in kept if c.verification == 'corrected')} corrected, {len(rejected)} rejected")
        return kept, rejected

    def _repair(self, nb: Notebook, messages: list[dict[str, str]], raw: list[LLMClaim],
                bad: list[tuple[LLMClaim, list[str]]]):
        """Send rejected claims back to the model once, with the citation errors.

        Returns (fixed, still rejected). Claims the model leaves out of its answer stay
        rejected. Repair is best effort: if the call fails, all claims stay rejected.
        """
        listing = "\n".join(
            f"- {json.dumps(c.model_dump())}\n  errors: {'; '.join(e)}" for c, e in bad
        )
        msgs = messages + [
            {"role": "assistant", "content": LLMClaimList(claims=raw).model_dump_json()},
            {"role": "user", "content": fill(load_prompt("repair_claims"), rejected=listing)},
        ]
        try:
            fixed = self.llm.structured(msgs, LLMClaimList, self.settings.model_extract, "claims").claims
        except Exception as e:  # repair is best effort
            log.warning("claim repair failed: %s", e)
            return [], bad
        ok, still_bad = validate_claims(nb, fixed)
        # Claims the model dropped during repair stay rejected with their original errors.
        fixed_statements = {c.statement for c, _ in ok} | {c.statement for c, _ in still_bad}
        dropped = [(c, e + ["dropped by the model during repair"]) for c, e in bad
                   if c.statement not in fixed_statements]
        return ok, still_bad + dropped


def extract_claims(nb: Notebook, settings: Settings, llm: LLMClient | None = None,
                   progress: Progress | None = None) -> ClaimSet:
    return ClaimExtractor(settings, llm or LLMClient(settings)).run(nb, progress=progress)


if __name__ == "__main__":  # python -m nb_expl.claims notebook.py [out.json]
    import sys

    from .config import get_settings
    from .parser import parse_file

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    nb = parse_file(sys.argv[1])
    result = extract_claims(nb, get_settings(), progress=print)
    out = json.dumps(result.model_dump(), indent=2)
    if len(sys.argv) > 2:
        with open(sys.argv[2], "w") as f:
            f.write(out)
    else:
        print(out)
