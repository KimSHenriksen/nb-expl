import json
from pathlib import Path

from fakes import fake_client

from nb_expl.claims import Citation, Claim, ClaimSet
from nb_expl.documents import (
    DOCUMENTS,
    DocumentWriter,
    LLMSection,
    SectionSpec,
    claims_for_section,
    github_line_link,
    render_markdown,
    validate_section,
)
from nb_expl.pipeline import run, save

EXAMPLE = Path(__file__).parent.parent / "examples" / "cash_application_engine.py"


def mk(id, topic="risk", importance="medium", refs=("C29:L361",), statement="s"):
    return Claim(id=id, source="llm", topic=topic, statement=statement, importance=importance,
                 certainty="explicit", verification="verified",
                 citations=[Citation(ref=r, quote="q") for r in refs])


CLAIMS = [
    mk("A001", topic="business_rule", refs=("C4:L33",), statement="TOL = 0.5"),
    mk("M001", topic="risk", importance="high", refs=("C29:L361", "C26:L333")),
    mk("M002", topic="data_flow", importance="low", refs=("C24:L271",)),
    mk("M003", topic="purpose", refs=("C1:L2-4",)),
]
SPEC = SectionSpec("risks", "What could go wrong", "x", ("risk",), include_high=True)


def test_claims_for_section_filters_by_topic_and_importance():
    spec = SectionSpec("s", "S", "x", ("business_rule",), include_high=True)
    assert [c.id for c in claims_for_section(spec, CLAIMS)] == ["A001", "M001"]  # code order
    spec = SectionSpec("s", "S", "x", ("business_rule", "risk", "purpose", "data_flow"), max_claims=2)
    assert {c.id for c in claims_for_section(spec, CLAIMS)} == {"M001", "M003"}  # high first, then code order


def test_validate_section_drops_uncited_facts_and_derives_refs():
    raw = LLMSection.model_validate({"blocks": [{"style": "bullets", "sentences": [
        {"text": "The tolerance becomes 5.", "kind": "fact", "claim_ids": ["M001", "A001"]},
        {"text": "Made-up fact.", "kind": "fact", "claim_ids": ["M999"]},
        {"text": "No ids at all.", "kind": "fact", "claim_ids": []},
        {"text": "In addition,", "kind": "transition", "claim_ids": ["M001"]},
        {"text": "Ask why TOL changes.", "kind": "guidance", "claim_ids": []},
    ]}]})
    blocks, dropped = validate_section(SPEC, raw, {c.id: c for c in CLAIMS})
    (b,) = blocks
    assert [s.text for s in b.sentences] == ["The tolerance becomes 5.", "In addition,", "Ask why TOL changes."]
    assert b.sentences[0].refs == ["C29:L361", "C26:L333", "C4:L33"]
    assert b.sentences[1].claim_ids == [] and b.sentences[1].refs == []
    assert [d.text for d in dropped] == ["Made-up fact.", "No ids at all."]
    assert "M999" in dropped[0].reason


def test_claim_ids_outside_the_section_are_not_accepted():
    raw = LLMSection.model_validate({"blocks": [{"style": "paragraph", "sentences": [
        {"text": "Data flows.", "kind": "fact", "claim_ids": ["M002"]}]}]})
    allowed = {c.id: c for c in claims_for_section(SPEC, CLAIMS)}  # M002 is not in this section
    blocks, dropped = validate_section(SPEC, raw, allowed)
    assert blocks == [] and len(dropped) == 1


def answer_for_any_section(kwargs):
    content = kwargs["messages"][-1]["content"]
    if "Section to write" in content:
        return json.dumps({"blocks": [{"style": "paragraph", "sentences": [
            {"text": "TOL is reassigned.", "kind": "fact", "claim_ids": ["A001"]},
            {"text": "Invented.", "kind": "fact", "claim_ids": ["ZZZ"]},
        ]}]})
    if "Claims to check" in content:
        return '{"verdicts": []}'
    return '{"claims": []}'


def test_writer_and_markdown(tmp_path):
    llm, comp, settings = fake_client([answer_for_any_section] * 50, tmp_path)
    cs = ClaimSet(claims=CLAIMS, rejected=[], stats={})
    doc = DocumentWriter(settings, llm).write("agent", cs, "nb.py")
    assert len(doc.sections) == len(DOCUMENTS["agent"].sections)
    assert sum(d.text == "Invented." for d in doc.dropped) == len(doc.sections)  # one per section
    md = render_markdown(doc, link=github_line_link("../nb.py"))
    assert "TOL is reassigned. [C4:L33](../nb.py#L33)" in md
    assert "Invented." not in md
    # the writer only ever sees claims chosen for its section
    prompts = [r["messages"][-1]["content"] for r in comp.requests]
    assert all("M002 [" not in p for p in prompts if "Risky areas" in p.split("Section to write:")[1][:20])


def test_failed_section_is_reported_not_fatal(tmp_path):
    llm, _, settings = fake_client(["not json"] * 50, tmp_path, structured_output="json_schema")
    cs = ClaimSet(claims=CLAIMS, rejected=[], stats={})
    doc = DocumentWriter(settings, llm).write("manager", cs, "nb.py")
    assert all(s.error for s in doc.sections)
    assert "could not be generated" in render_markdown(doc)


def test_pipeline_end_to_end_with_fake_llm(tmp_path):
    llm, _, settings = fake_client([answer_for_any_section] * 200, tmp_path / "cache", chunk_lines=10_000)
    nb, result = run(EXAMPLE, settings=settings, llm=llm)
    assert set(result.documents) == {"manager", "developer", "agent"}
    out = tmp_path / "out"
    files = save(result, out)
    assert {p.name for p in files} == {"manager.md", "developer.md", "agent.md", "run.json"}
    md = (out / "developer.md").read_text()
    # links are relative from the output directory to the notebook file
    assert "cash_application_engine.py#L" in md and "](/" not in md
    data = json.loads((out / "run.json").read_text())
    assert data["claims"]["stats"]["analysis_claims"] > 100


def test_merge_refs_drops_contained_ranges():
    from nb_expl.documents import merge_refs
    assert merge_refs(["C37:L459", "C37:L458-459", "C38:L474", "C37:L459"]) == ["C37:L458-459", "C38:L474"]
    assert merge_refs(["C26:L333", "C26:L333-335", "C4:L33"]) == ["C26:L333-335", "C4:L33"]


def test_high_static_facts_reach_every_section():
    static = mk("A050", topic="risk", importance="high", refs=("C29:L361",))
    static = static.model_copy(update={"source": "analysis", "verification": "static"})
    spec = SectionSpec("s", "S", "x", ("purpose",))
    assert "A050" in {c.id for c in claims_for_section(spec, CLAIMS + [static])}
