"""The three audience documents, written by the LLM from validated claims only.

Each document is a fixed list of sections. Every section is written in its own
LLM call, which sees only the claims relevant to that section. The model
returns sentences tagged with the claim ids that support them; we then:

- drop claim ids that do not exist,
- delete "fact" sentences that end up with no supporting claim,
- turn claim ids into code citations (C<cell>:L<lines>) when rendering.

So every factual sentence in the output is traceable to code, and the code
behind each citation has already been checked mechanically.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Literal

from pydantic import BaseModel, ConfigDict

from .claims import Claim, ClaimSet, Topic
from .config import Settings
from .llm import LLMClient
from .utils import Progress, chat_messages, fill, load_prompt, parallel_map, reporter
from .refs import Ref, parse_ref

log = logging.getLogger(__name__)
Audience = Literal["manager", "developer", "agent"]

# --- document structure ---------------------------------------------------------


@dataclass(frozen=True)
class SectionSpec:
    key: str
    title: str
    instructions: str
    topics: tuple[Topic, ...]
    include_high: bool = False
    """Also give the writer every high-importance claim, whatever its topic."""
    max_claims: int = 90


@dataclass(frozen=True)
class DocumentSpec:
    audience: Audience
    title: str
    reader: str
    sections: tuple[SectionSpec, ...]


DOCUMENTS: dict[Audience, DocumentSpec] = {
    "manager": DocumentSpec(
        "manager", "Overview for managers",
        "A non-technical manager who owns the business process this notebook supports. They do not read "
        "code. Explain in the business terms of the notebook's own domain, as the claims reveal it. Avoid "
        "variable names unless a business user would recognise them (table names and parameters are fine).",
        (
            SectionSpec("summary", "In short",
                        "A short paragraph (3-5 sentences): what the notebook is for, what it produces, "
                        "and how it is run.", ("purpose", "output", "parameter"), max_claims=40),
            SectionSpec("inputs_outputs", "What goes in and what comes out",
                        "Two bullet lists: the input data it reads and the results it produces "
                        "(tables, reports, notifications), in business terms.",
                        ("input", "output", "side_effect", "parameter")),
            SectionSpec("how_it_works", "How it works",
                        "Explain, in order, the main steps that turn the inputs into the results: which "
                        "records are selected, how they are matched, combined or calculated, which thresholds "
                        "and tolerances apply, and what happens to records that do not fit. A paragraph "
                        "followed by a numbered-style bullet list.",
                        ("business_rule", "calculation", "filter", "purpose")),
            SectionSpec("risks", "What could go wrong",
                        "A bullet list of the most important business risks: wrong results, records "
                        "silently left out or duplicated, data overwritten or lost, values that differ from "
                        "what people expect, surprises in test or dry-run modes. For each, say what happens "
                        "and why it matters. Most important first.",
                        ("risk", "data_quality", "filter", "side_effect", "invariant"), include_high=True),
            SectionSpec("questions", "Questions to ask the team",
                        "A short bullet list of guidance sentences: concrete questions a manager should ask "
                        "the people who maintain this notebook, each tied to a risk above.",
                        ("risk", "dead_code", "dependency"), include_high=True, max_claims=50),
        ),
    ),
    "developer": DocumentSpec(
        "developer", "Guide for developers",
        "A developer who is new to this code and has to maintain it. They read Python and Spark. Use the "
        "real variable, column, function and table names, and explain what the cryptic names mean.",
        (
            SectionSpec("overview", "Overview",
                        "One paragraph: what the notebook does technically, the main stages, and the "
                        "run parameters (widgets).", ("purpose", "parameter", "structure", "output"),
                        max_claims=60),
            SectionSpec("structure", "Structure, cell by cell",
                        "A bullet list walking through the cells in order, grouping neighbouring cells "
                        "that belong together. Mention dead and commented-out code where it sits.",
                        ("purpose", "structure", "dead_code", "dependency"), max_claims=110),
            SectionSpec("data_flow", "How data flows",
                        "Describe how data moves from the inputs through the main variables (DataFrames, "
                        "tables, collections) to the outputs, including where it changes form, for example "
                        "from Spark to pandas. A paragraph, then a bullet list of the main variables and "
                        "what each holds.",
                        ("data_flow", "input", "output"), max_claims=120),
            SectionSpec("rules", "Rules in the code",
                        "Bullet list of the rules the code applies, in execution order: filters, joins and "
                        "matching, calculations, thresholds and tolerances with their actual values, "
                        "conversions, special cases and hard-coded exceptions.",
                        ("business_rule", "calculation", "filter")),
            SectionSpec("outputs", "Outputs, side effects and dependencies",
                        "What is written where and in which mode, what runs in a dry run or test mode and "
                        "what does not, and what the notebook needs from outside (%run notebooks, Databricks "
                        "globals, external services).",
                        ("output", "side_effect", "dependency", "parameter")),
            SectionSpec("pitfalls", "Pitfalls",
                        "Bullet list of things in the code that will surprise a developer: misleading names, "
                        "reassigned constants, global mutable state, silent data loss, checks that cannot "
                        "fail, non-determinism.", ("risk", "data_quality", "dead_code"), include_high=True),
        ),
    ),
    "agent": DocumentSpec(
        "agent", "Brief for an AI coding agent",
        "An AI coding agent that will debug or change this notebook. Be precise and terse; prefer bullet "
        "lists; use exact names, values and line-level facts. State what must stay true and how to check it.",
        (
            SectionSpec("scope", "Scope and entry points",
                        "Bullets: parameters (widgets) and their defaults, input files, output tables with "
                        "write modes, external dependencies.",
                        ("parameter", "input", "output", "dependency"), max_claims=70),
            SectionSpec("invariants", "Invariants",
                        "Bullets: properties that must hold before and after any change (totals that must "
                        "reconcile, uniqueness, ordering, which steps run under which conditions, values of "
                        "constants in effect). Say for each whether the code checks it, and how.",
                        ("invariant", "business_rule", "calculation"), include_high=True),
            SectionSpec("risky_areas", "Risky areas",
                        "Bullets: the code regions where a change is most likely to break something or "
                        "already behaves unexpectedly, and why.",
                        ("risk", "side_effect", "data_quality", "filter"), include_high=True),
            SectionSpec("state", "Hidden state and ordering",
                        "Bullets: global mutable state, functions that read or mutate globals, values "
                        "reassigned between cells, and execution-order dependence.",
                        ("risk", "structure", "data_flow"), max_claims=80),
            SectionSpec("checks", "How to check that a change is safe",
                        "Guidance bullets: concrete checks to run before and after a change (dry-run or test "
                        "modes if the notebook has them, outputs and totals to compare, row counts, which "
                        "tables are touched), each tied to the claims about the code it protects.",
                        ("invariant", "output", "side_effect", "parameter", "data_quality"), include_high=True),
        ),
    ),
}

# --- what the LLM returns ---------------------------------------------------------


class LLMSentence(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str
    kind: Literal["fact", "guidance", "transition"]
    claim_ids: list[str]


class LLMBlock(BaseModel):
    model_config = ConfigDict(extra="forbid")
    style: Literal["paragraph", "bullets"]
    sentences: list[LLMSentence]


class LLMSection(BaseModel):
    model_config = ConfigDict(extra="forbid")
    blocks: list[LLMBlock]


# --- validated documents ----------------------------------------------------------


class Sentence(BaseModel):
    text: str
    kind: Literal["fact", "guidance", "transition"]
    claim_ids: list[str]
    refs: list[str]
    """Code citations, derived from the claims (deduplicated, in order)."""


class Block(BaseModel):
    style: Literal["paragraph", "bullets"]
    sentences: list[Sentence]


class DroppedSentence(BaseModel):
    section: str
    text: str
    reason: str


class Section(BaseModel):
    key: str
    title: str
    blocks: list[Block]
    error: str | None = None


class Document(BaseModel):
    audience: Audience
    title: str
    notebook: str
    sections: list[Section]
    dropped: list[DroppedSentence]
    stats: dict[str, int]


# --- generation -------------------------------------------------------------------

_IMPORTANCE_ORDER = {"high": 0, "medium": 1, "low": 2}


def claims_for_section(spec: SectionSpec, claims: list[Claim]) -> list[Claim]:
    """Claims on the section's topics, capped by importance. High-importance static facts
    (reassigned constants, values in effect, writes, unguarded side effects) are always
    added, so no section can contradict them by quoting an outdated value."""
    chosen = [c for c in claims if c.topic in spec.topics or (spec.include_high and c.importance == "high")]
    chosen.sort(key=lambda c: (_IMPORTANCE_ORDER[c.importance], _first_line(c)))
    chosen = chosen[: spec.max_claims]
    ids = {c.id for c in chosen}
    chosen += [c for c in claims if c.source == "analysis" and c.importance == "high" and c.id not in ids]
    chosen.sort(key=_first_line)  # present in code order
    return chosen


def merge_refs(refs: list[str]) -> list[str]:
    """Drop refs contained in another ref of the list (C37:L459 inside C37:L458-459), keep order."""
    parsed = [parse_ref(r) for r in dict.fromkeys(refs)]
    keep = [r for r in parsed if not any(o != r and o.cell == r.cell and o.start <= r.start and r.end <= o.end
                                         for o in parsed)]
    return [str(r) for r in keep]


def _first_line(c: Claim) -> int:
    return min(parse_ref(x.ref).start for x in c.citations)


def render_claims(claims: list[Claim]) -> str:
    """Claims as one line each, the form in which the writer sees them."""
    return "\n".join(f"{c.id} [{c.topic}, {c.importance}, {c.verification}] {c.statement}" for c in claims)


def validate_section(spec: SectionSpec, raw: LLMSection, known: dict[str, Claim]
                     ) -> tuple[list[Block], list[DroppedSentence]]:
    """Turn the model's section into validated blocks.

    This is where "every factual sentence cites its source" is enforced: a fact
    sentence without a known claim id is dropped and reported. A sentence's refs come
    from its claims' citations, never from the model.
    """
    blocks, dropped = [], []
    for b in raw.blocks:
        sentences = []
        for s in b.sentences:
            text = s.text.strip()
            if not text:
                continue
            ids = list(dict.fromkeys(i.strip() for i in s.claim_ids if i.strip() in known))
            unknown = [i for i in s.claim_ids if i.strip() not in known]
            if s.kind == "fact" and not ids:
                reason = "fact without a valid claim id" + (f" (unknown ids: {unknown})" if unknown else "")
                dropped.append(DroppedSentence(section=spec.key, text=text, reason=reason))
                continue
            if s.kind == "transition":
                ids = []
            refs = merge_refs([cit.ref for i in ids for cit in known[i].citations])
            sentences.append(Sentence(text=text, kind=s.kind, claim_ids=ids, refs=refs))
        if sentences:
            blocks.append(Block(style=b.style, sentences=sentences))
    return blocks, dropped


class DocumentWriter:
    def __init__(self, settings: Settings, llm: LLMClient):
        self.settings = settings
        self.llm = llm
        self.system = load_prompt("write_system")

    def write(self, audience: Audience, claim_set: ClaimSet, notebook_name: str,
              progress: Progress | None = None) -> Document:
        """Write every section of one document. A failed section is kept with its error
        instead of failing the document.
        """
        say = reporter(progress, log)
        spec = DOCUMENTS[audience]
        known = claim_set.by_id()

        def work(section: SectionSpec) -> tuple[Section, list[DroppedSentence]]:
            """Write and validate one section from the claims selected for it."""
            say(f"writing {audience}: {section.title}")
            claims = claims_for_section(section, claim_set.claims)
            outline = "\n".join(f"{'>>' if s is section else '  '} {s.title}" for s in spec.sections)
            prompt = fill(load_prompt("write_section"), document=f"{spec.title}: {notebook_name}",
                          audience=spec.reader, outline=outline, section=section.title,
                          instructions=section.instructions, claims=render_claims(claims))
            messages = chat_messages(self.system, prompt)
            try:
                raw = self.llm.structured(messages, LLMSection, self.settings.model_write, "section")
            except Exception as e:
                log.exception("writing section %s/%s failed", audience, section.key)
                say(f"{audience}: {section.title} FAILED ({e})")
                return Section(key=section.key, title=section.title, blocks=[], error=str(e)), []
            blocks, dropped = validate_section(section, raw, {c.id: known[c.id] for c in claims})
            return Section(key=section.key, title=section.title, blocks=blocks), dropped

        results = parallel_map(work, spec.sections, self.settings.concurrency)

        sections = [s for s, _ in results]
        dropped = [d for _, ds in results for d in ds]
        sentences = [s for sec in sections for b in sec.blocks for s in b.sentences]
        stats = {
            "sentences": len(sentences),
            "fact_sentences": sum(1 for s in sentences if s.kind == "fact"),
            "guidance_sentences": sum(1 for s in sentences if s.kind == "guidance"),
            "dropped_sentences": len(dropped),
            "claims_cited": len({i for s in sentences for i in s.claim_ids}),
            "failed_sections": sum(1 for s in sections if s.error),
        }
        say(f"{audience}: {stats}")
        return Document(audience=audience, title=spec.title, notebook=notebook_name,
                        sections=sections, dropped=dropped, stats=stats)


# --- rendering ----------------------------------------------------------------------


def render_markdown(doc: Document, link: Callable[[Ref], str] | None = None, max_refs: int = 4) -> str:
    """Markdown with inline citations. `link` maps a ref to a URL (e.g. the notebook file
    with a line anchor); without it citations are plain text."""

    def cite(refs: list[str]) -> str:
        """Citation links for a sentence: at most max_refs, then "+N"."""
        if not refs:
            return ""
        shown = refs[:max_refs]
        parts = []
        for r in shown:
            parts.append(f"[{r}]({link(parse_ref(r))})" if link else f"[{r}]")
        more = f" +{len(refs) - max_refs}" if len(refs) > max_refs else ""
        return " " + " ".join(parts) + more

    out = [f"# {doc.title}: `{doc.notebook}`", ""]
    out.append("_Generated from the notebook's code. Every factual sentence cites the cell and lines it "
               "comes from (`C<cell>:L<lines>`); sentences in italics are guidance, not facts._")
    out.append("")
    for sec in doc.sections:
        out += [f"## {sec.title}", ""]
        if sec.error:
            out += [f"> This section could not be generated: {sec.error}", ""]
            continue
        for b in sec.blocks:
            rendered = [_sentence_md(s) + cite(s.refs) for s in b.sentences]
            if b.style == "bullets":
                out += [f"- {r}" for r in rendered]
            else:
                out.append(" ".join(rendered))
            out.append("")
    return "\n".join(out).rstrip() + "\n"


def _sentence_md(s: Sentence) -> str:
    """Guidance sentences in italics, so readers can tell them from cited facts."""
    return f"_{s.text}_" if s.kind == "guidance" else s.text


def github_line_link(notebook_path: str) -> Callable[[Ref], str]:
    """Links like ../notebook.py#L57-L60, which GitHub and Forgejo render as highlighted lines."""
    def link(ref: Ref) -> str:
        anchor = f"#L{ref.start}" if ref.start == ref.end else f"#L{ref.start}-L{ref.end}"
        return f"{notebook_path}{anchor}"
    return link
