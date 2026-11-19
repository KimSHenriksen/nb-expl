import json
from pathlib import Path

import pytest

from nb_expl.claims import (
    ClaimExtractor,
    LLMClaimList,
    clean_quote,
    find_quote,
    validate_citation,
)
from nb_expl.llm import LLMError, extract_json, strict_schema
from nb_expl.parser import parse_file
from nb_expl.refs import Ref

EXAMPLE = Path(__file__).parent.parent / "examples" / "cash_application_engine.py"


@pytest.fixture(scope="module")
def nb():
    return parse_file(EXAMPLE)


from fakes import bad_request, fake_client  # noqa: E402


# --- citation validation --------------------------------------------------------


def test_exact_citation_is_valid(nb):
    c, err = validate_citation(nb, "C4:L33", "TOL = 0.5")
    assert err is None and c.ref == "C4:L33" and c.note is None


def test_quote_with_gutter_and_quotes_is_cleaned():
    assert clean_quote("`361| TOL = 5.0`") == "TOL = 5.0"
    assert clean_quote("  ds = sorted(ds,\n   key=lambda d: -bal[d]) ") == "ds = sorted(ds, key=lambda d: -bal[d])"


def test_double_escaped_quote_is_unescaped(nb):
    c, err = validate_citation(nb, "C14:L138", r'rr.withColumn(\"id_src\", F.when(F.col(\"c_ov\").isNotNull()')
    assert err is None and c.ref == "C14:L138"
    c, err = validate_citation(nb, "C1:L3-4", r"## cash app v3\nnew logic from Q2")
    assert err is None and c.ref == "C1:L3-4"


def test_multiline_quote(nb):
    c, err = validate_citation(nb, "C26:L319-320", "bal[d] > 0]\n    ds = sorted(ds, key=lambda d: -bal[d])")
    assert err is None and c.ref == "C26:L319-320"


def test_wrong_cell_number_is_corrected(nb):
    c, err = validate_citation(nb, "C5:L361", "TOL = 5.0")
    assert err is None and c.ref == "C29:L361" and c.note == "cell number corrected"


def test_lines_slightly_off_are_adjusted(nb):
    c, err = validate_citation(nb, "C26:L318", "ds = sorted(ds, key=lambda d: -bal[d])")
    assert err is None and c.ref == "C26:L320" and "adjusted" in c.note


def test_unique_quote_elsewhere_is_relocated(nb):
    c, err = validate_citation(nb, "C2:L10", 'notify_ar(h)')
    assert err is None and c.ref == "C43:L532" and "relocated" in c.note


def test_invented_quote_is_rejected(nb):
    c, err = validate_citation(nb, "C26:L320", "sorted(ds, key=lambda d: meta[d]['due_dt'])")
    assert c is None and "does not appear" in err


def test_ambiguous_quote_far_away_is_rejected(nb):
    c, err = validate_citation(nb, "C2:L10", "bal[d] = bal[d] - a")
    assert c is None and "other places" in err


def test_find_quote_returns_minimal_windows(nb):
    assert find_quote(nb, "TOL = 5.0") == [Ref(29, 361, 361)]


def test_magic_cells_match_without_prefix(nb):
    c, err = validate_citation(nb, "C5:L44", "%run ../shared/fin_utils")
    assert err is None and c.ref == "C5:L44"


# --- LLM client -------------------------------------------------------------------


def test_extract_json_tolerates_noise():
    assert extract_json('<think>hmm</think>\n```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_json('Sure! {"a": [1, 2]} hope this helps') == {"a": [1, 2]}
    with pytest.raises(ValueError):
        extract_json("no json here")


def test_strict_schema_closes_objects():
    s = strict_schema(LLMClaimList)
    claim = s["$defs"]["LLMClaim"]
    assert claim["additionalProperties"] is False
    assert set(claim["required"]) == {"topic", "statement", "importance", "certainty", "citations"}


def test_falls_back_when_json_schema_is_rejected(tmp_path):
    llm, comp, _ = fake_client([bad_request(), '{"claims": []}'], tmp_path)
    out = llm.structured([{"role": "user", "content": "x"}], LLMClaimList, "m", "claims")
    assert out.claims == []
    assert comp.requests[0]["response_format"]["type"] == "json_schema"
    assert comp.requests[1]["response_format"] == {"type": "json_object"}
    assert "JSON Schema" in comp.requests[1]["messages"][-1]["content"]
    # the working mode is remembered
    llm.structured([{"role": "user", "content": "y"}], LLMClaimList, "m", "claims")
    assert comp.requests[2]["response_format"] == {"type": "json_object"}


def test_invalid_json_is_sent_back_once(tmp_path):
    llm, comp, _ = fake_client(['{"claims": [{"topic": "nonsense"}]}', '{"claims": []}'], tmp_path)
    llm.structured([{"role": "user", "content": "x"}], LLMClaimList, "m", "claims")
    assert len(comp.requests) == 2
    assert "invalid" in comp.requests[1]["messages"][-1]["content"]


def test_gives_up_after_repair_attempts(tmp_path):
    llm, _, _ = fake_client(["nope", "still nope"], tmp_path, structured_output="json_schema")
    with pytest.raises(LLMError):
        llm.structured([{"role": "user", "content": "x"}], LLMClaimList, "m", "claims")


def test_truncation_by_hidden_reasoning_is_explained(tmp_path):
    llm, _, _ = fake_client([('{"cla', "length", "x" * 5000)], tmp_path)
    with pytest.raises(LLMError, match="LLM_EXTRA_BODY"):
        llm.structured([{"role": "user", "content": "x"}], LLMClaimList, "m", "claims")


def test_empty_answer_is_an_error(tmp_path):
    llm, _, _ = fake_client([("", "stop", "thinking...")], tmp_path)
    with pytest.raises(LLMError, match="empty"):
        llm.structured([{"role": "user", "content": "x"}], LLMClaimList, "m", "claims")


def test_extra_body_is_sent(tmp_path):
    body = {"chat_template_kwargs": {"enable_thinking": False}}
    llm, comp, _ = fake_client(['{"claims": []}'], tmp_path, llm_extra_body=body)
    llm.structured([{"role": "user", "content": "x"}], LLMClaimList, "m", "claims")
    assert comp.requests[0]["extra_body"] == body


def test_invalid_answers_are_not_cached(tmp_path):
    llm, comp, _ = fake_client(["junk", "junk"], tmp_path, structured_output="json_schema")
    with pytest.raises(LLMError):
        llm.structured([{"role": "user", "content": "x"}], LLMClaimList, "m", "claims")
    assert not list(tmp_path.rglob("*.json"))


def test_failed_chunk_does_not_abort_extraction(nb, tmp_path):
    llm, _, settings = fake_client(['{"claims": []}'] * 20, tmp_path, structured_output="json_schema")
    llm.structured = _fail_first(llm.structured)
    result = ClaimExtractor(settings, llm).run(nb)
    assert result.stats["failed_jobs"] == 1 and "boom" in result.errors[0]
    assert result.stats["analysis_claims"] > 100


def _fail_first(fn):
    state = {"n": 0}

    def wrapper(*a, **kw):
        state["n"] += 1
        if state["n"] == 1:
            raise LLMError("boom")
        return fn(*a, **kw)

    return wrapper


def test_cache_avoids_second_call(tmp_path):
    llm, comp, _ = fake_client(['{"claims": []}'], tmp_path)
    msgs = [{"role": "user", "content": "x"}]
    llm.structured(msgs, LLMClaimList, "m", "claims")
    llm.structured(msgs, LLMClaimList, "m", "claims")
    assert len(comp.requests) == 1 and llm.cache_hits == 1


# --- extraction end to end with a fake model ----------------------------------------


def claim(statement, ref, quote, topic="business_rule"):
    return {"topic": topic, "statement": statement, "importance": "high", "certainty": "explicit",
            "citations": [{"ref": ref, "quote": quote}]}


def test_extractor_validates_repairs_and_rejects(nb, tmp_path):
    good = claim("FIFO sorts by largest balance", "C26:L320", "key=lambda d: -bal[d]")
    bad = claim("Receipts are sorted by due date", "C26:L320", "sorted by due_dt")
    fixed = claim("Receipts are processed by value date", "C30:L367", 'rp.sort_values(["val_dt", "rcpt_id"])')

    def answer(kwargs):
        content = kwargs["messages"][-1]["content"]
        if "were rejected" in content:
            return json.dumps({"claims": [fixed]})
        if "Full notebook" in content:
            return json.dumps({"claims": [good, bad]})
        return '{"claims": []}'

    llm, comp, settings = fake_client([answer] * 20, tmp_path, chunk_lines=10_000, verify_claims=False)
    result = ClaimExtractor(settings, llm).run(nb)

    llm_claims = [c for c in result.claims if c.source == "llm"]
    assert [c.statement for c in llm_claims] == ["FIFO sorts by largest balance",
                                                  "Receipts are processed by value date"]
    assert llm_claims[0].id == "M001"
    # The model replaced the bad claim with a reworded one, so the original stays on record.
    (rej,) = result.rejected
    assert rej.statement == "Receipts are sorted by due date"
    assert "does not appear" in rej.errors[0]
    assert result.stats["analysis_claims"] > 100
    assert all(c.citations for c in result.claims)


def test_verification_keeps_corrects_and_rejects(nb, tmp_path):
    claims = [
        claim("FIFO sorts by largest balance first", "C26:L320", "key=lambda d: -bal[d]"),
        claim("Allocation lines are written when dry is not 'N'", "C37:L458-459", 'if dry == "N":'),
        claim("Write-offs are capped by MAX_WO", "C26:L333", "bal[d] <= TOL"),
    ]
    seen = {}

    def answer(kwargs):
        content = kwargs["messages"][-1]["content"]
        if "Claims to check" in content:
            seen["prompt"] = content
            return json.dumps({"verdicts": [
                {"id": "M001", "code_says": "x", "verdict": "supported", "reason": "sorted by -bal[d]", "corrected_statement": ""},
                {"id": "M002", "code_says": "x", "verdict": "partially_supported", "reason": "condition is inverted",
                 "corrected_statement": "Allocation lines are written only when dry == 'N'"},
                {"id": "M003", "code_says": "x", "verdict": "unsupported", "reason": "MAX_WO is never used", "corrected_statement": ""},
            ]})
        if "Full notebook" in content:
            return json.dumps({"claims": claims})
        return '{"claims": []}'

    llm, _, settings = fake_client([answer] * 20, tmp_path, chunk_lines=10_000)
    result = ClaimExtractor(settings, llm).run(nb)
    by_id = {c.id: c for c in result.claims if c.source == "llm"}

    assert by_id["M001"].verification == "verified"
    assert by_id["M002"].verification == "corrected"
    assert by_id["M002"].statement == "Allocation lines are written only when dry == 'N'"
    assert by_id["M002"].original_statement.startswith("Allocation lines are written when dry is not")
    assert "M003" not in by_id
    (rej,) = [r for r in result.rejected if r.stage == "verification"]
    assert "MAX_WO is never used" in rej.errors[0]
    assert result.stats["verified"] == 1 and result.stats["corrected"] == 1
    # analysis facts are not sent for verification and stay "static"
    assert all(c.verification == "static" for c in result.claims if c.source == "analysis")
    # the reviewer sees the cited lines marked and surrounding context
    assert '>458| if dry == "N":' in seen["prompt"]
    assert "## constant_reassigned" in seen["prompt"]  # fact sheet included
    assert " 460|     print(\"written\"" in seen["prompt"]  # context after, within the cell
    assert " 456| " not in seen["prompt"]  # context never crosses into another cell


def test_failed_verification_leaves_claims_unverified(nb, tmp_path):
    good = claim("FIFO sorts by largest balance first", "C26:L320", "key=lambda d: -bal[d]")

    def answer(kwargs):
        content = kwargs["messages"][-1]["content"]
        if "Full notebook" in content:
            return json.dumps({"claims": [good]})
        if "Claims to check" in content or "invalid" in content:
            return "not json"
        return '{"claims": []}'

    llm, _, settings = fake_client([answer] * 20, tmp_path, chunk_lines=10_000,
                                   structured_output="json_schema")
    result = ClaimExtractor(settings, llm).run(nb)
    (m,) = [c for c in result.claims if c.source == "llm"]
    assert m.verification == "unverified"
    assert any(e.startswith("verification") for e in result.errors)
