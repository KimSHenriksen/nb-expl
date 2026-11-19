import json
import re
import time
from pathlib import Path

import pytest
from fakes import fake_client
from fastapi.testclient import TestClient

from nb_expl.app import create_app

EXAMPLE = Path(__file__).parent.parent / "examples" / "cash_application_engine.py"


def answer(kwargs):
    content = kwargs["messages"][-1]["content"]
    if "Section to write" in content:
        first_id = re.search(r"^([AM]\d{3}) \[", content.split("# CLAIMS")[1], re.M).group(1)
        return json.dumps({"blocks": [{"style": "paragraph", "sentences": [
            {"text": "It has widgets.", "kind": "fact", "claim_ids": [first_id]}]}]})
    return '{"claims": []}'


@pytest.fixture
def client(tmp_path):
    llm, _, settings = fake_client([answer] * 500, tmp_path, chunk_lines=10_000)
    return TestClient(create_app(settings, llm=llm))


def wait_done(client, run_id, timeout=20):
    deadline = time.time() + timeout
    while time.time() < deadline:
        run = client.get(f"/api/runs/{run_id}").json()
        if run["status"] != "running":
            return run
        time.sleep(0.05)
    raise AssertionError("run did not finish")


def test_index_and_static(client):
    assert "Notebook Explainer" in client.get("/").text
    assert client.get("/static/app.js").status_code == 200


def test_config_never_exposes_api_key(client):
    cfg = client.get("/api/config").json()
    assert cfg["example_path"].endswith("cash_application_engine.py")
    assert "key" not in str(cfg).lower()


def test_run_by_path_end_to_end(client):
    run_id = client.post("/api/runs", json={"path": str(EXAMPLE)}).json()["run_id"]
    run = wait_done(client, run_id)
    assert run["status"] == "done", run["error"]
    assert set(run["documents"]) == {"manager", "developer", "agent"}
    assert run["notebook"]["lines"][32] == "TOL = 0.5"  # file line 33
    assert run["notebook"]["cells"][3]["start_line"] == 33
    sentence = run["documents"]["manager"]["sections"][0]["blocks"][0]["sentences"][0]
    assert sentence["refs"] and len(sentence["claim_ids"]) == 1
    md = client.get(f"/api/runs/{run_id}/manager.md").text
    assert "It has widgets." in md


def test_events_stream_ends_with_done(client):
    run_id = client.post("/api/runs", json={"path": str(EXAMPLE)}).json()["run_id"]
    body = client.get(f"/api/runs/{run_id}/events").text
    assert "parsing" in body
    assert body.rstrip().endswith('data: {"error": null}')
    assert "event: done" in body


def test_upload(client):
    with open(EXAMPLE, "rb") as f:
        r = client.post("/api/runs/upload", files={"file": ("nb.py", f, "text/x-python")})
    run = wait_done(client, r.json()["run_id"])
    assert run["status"] == "done" and run["notebook"]["name"] == "nb.py"


def test_bad_inputs_fail_fast(client, tmp_path):
    assert client.post("/api/runs", json={"path": str(tmp_path / "nope.py")}).status_code == 404
    assert client.post("/api/runs", json={"path": str(tmp_path)}).status_code == 400
    plain = tmp_path / "plain.py"
    plain.write_text("print('hi')\n")
    r = client.post("/api/runs", json={"path": str(plain)})
    assert r.status_code == 400 and "Databricks" in r.json()["detail"]
    assert client.get("/api/runs/unknown").status_code == 404


def test_failed_run_is_reported(tmp_path):
    llm, _, settings = fake_client([], tmp_path)
    llm.structured = None  # any LLM use raises TypeError
    app_client = TestClient(create_app(settings, llm=llm))
    run_id = app_client.post("/api/runs", json={"path": str(EXAMPLE)}).json()["run_id"]
    run = wait_done(app_client, run_id)
    # extraction failures are per chunk, so the run still completes, with errors recorded
    assert run["status"] == "done"
    assert run["claims"]["errors"]
