"""Local web app: point it at a notebook on disk, watch progress, read the cited documents.

Start with `uv run nb-expl`. The server binds to 127.0.0.1 only: reading
arbitrary local files is the point of the app, so it must not be exposed.
"""

from __future__ import annotations

import json
import threading
import time
import uuid
import webbrowser
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Any, Iterator

import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, PlainTextResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .config import Settings, get_settings
from .documents import render_markdown
from .llm import LLMClient
from .parser import Notebook, NotebookFormatError, parse_file
from .pipeline import RunResult, run

STATIC = Path(str(resources.files("nb_expl").joinpath("static")))
EXAMPLE = Path(__file__).resolve().parents[2] / "examples" / "cash_application_engine.py"


@dataclass
class Job:
    id: str
    path: str
    status: str = "running"  # running | done | error
    events: list[str] = field(default_factory=list)
    error: str | None = None
    notebook: Notebook | None = None
    result: RunResult | None = None
    started: float = field(default_factory=time.time)
    cond: threading.Condition = field(default_factory=threading.Condition)

    def emit(self, message: str) -> None:
        """Add a progress line and wake every stream waiting for it."""
        with self.cond:
            self.events.append(message)
            self.cond.notify_all()

    def finish(self, status: str, error: str | None = None) -> None:
        """Mark the job done or failed and wake waiting streams so they can close."""
        with self.cond:
            self.status, self.error = status, error
            self.cond.notify_all()


class RunRequest(BaseModel):
    path: str


def create_app(settings: Settings | None = None, llm: LLMClient | None = None) -> FastAPI:
    """The FastAPI app. Runs are kept in memory and executed in background threads;
    `llm` lets tests inject a fake client.
    """
    settings = settings or get_settings()
    jobs: dict[str, Job] = {}
    shared_llm = llm  # tests inject a fake; otherwise one client per run
    app = FastAPI(title="nb-expl")

    def start(path: Path) -> Job:
        """Check that the file is a notebook, then start the pipeline in a background thread.

        The parse happens up front so bad paths fail with a clear HTTP error instead of
        after a long run.
        """
        try:
            parse_file(path)  # fail fast, before starting a long run
        except FileNotFoundError:
            raise HTTPException(404, f"File not found: {path}")
        except IsADirectoryError:
            raise HTTPException(400, f"That is a directory, not a file: {path}")
        except (NotebookFormatError, UnicodeDecodeError) as e:
            raise HTTPException(400, f"Not a Databricks notebook source file: {e}")
        job = Job(id=uuid.uuid4().hex[:12], path=str(path))
        jobs[job.id] = job

        def work() -> None:
            try:
                nb, result = run(path, settings=settings, progress=job.emit,
                                 llm=shared_llm or LLMClient(settings))
                job.notebook, job.result = nb, result
                job.finish("done")
            except Exception as e:  # report any failure to the browser
                job.emit(f"FAILED: {e}")
                job.finish("error", str(e))

        threading.Thread(target=work, daemon=True).start()
        return job

    def get_job(run_id: str) -> Job:
        """The job for run_id, or HTTP 404."""
        if run_id not in jobs:
            raise HTTPException(404, "Unknown run")
        return jobs[run_id]

    @app.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/config")
    def config() -> dict[str, Any]:
        """What the UI needs to know; never includes the API key."""
        return {
            "example_path": str(EXAMPLE) if EXAMPLE.exists() else "",
            "model_extract": settings.model_extract,
            "model_write": settings.model_write,
            "model_verify": settings.model_verify or settings.model_extract,
            "verify_claims": settings.verify_claims,
        }

    @app.post("/api/runs")
    def create_run(req: RunRequest) -> dict[str, str]:
        """Start a run for a path on the server's disk; quotes around the path and ~ are allowed."""
        path = Path(req.path.strip().strip('"').strip("'")).expanduser()
        if not path.is_absolute():
            path = path.resolve()
        return {"run_id": start(path).id}

    @app.post("/api/runs/upload")
    async def upload_run(file: UploadFile = File(...)) -> dict[str, str]:
        """Save an uploaded notebook under cache_dir/uploads and start a run for it."""
        data = await file.read()
        name = Path(file.filename or "notebook.py").name
        dest = Path(settings.cache_dir) / "uploads" / uuid.uuid4().hex[:8] / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        return {"run_id": start(dest.resolve()).id}

    @app.get("/api/runs/{run_id}/events")
    def events(run_id: str) -> StreamingResponse:
        """Server-sent events: one `data:` line per progress message, a keep-alive comment
        every 15 seconds, and a final `event: done` or `event: error`.
        """
        job = get_job(run_id)

        def stream() -> Iterator[str]:
            sent = 0
            while True:
                with job.cond:
                    while sent == len(job.events) and job.status == "running":
                        job.cond.wait(timeout=15)
                        if sent == len(job.events) and job.status == "running":
                            break  # heartbeat
                    new = job.events[sent:]
                    status = job.status
                sent += len(new)
                for msg in new:
                    yield f"data: {json.dumps({'message': msg})}\n\n"
                if not new and status == "running":
                    yield ": keep-alive\n\n"
                if status != "running" and sent == len(job.events):
                    yield f"event: {status}\ndata: {json.dumps({'error': job.error})}\n\n"
                    return

        return StreamingResponse(stream(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @app.get("/api/runs/{run_id}")
    def get_run(run_id: str) -> dict[str, Any]:
        """Status and events; once done, also the notebook lines and cells (for the code
        panel), the claims and the documents.
        """
        job = get_job(run_id)
        out: dict[str, Any] = {"id": job.id, "path": job.path, "status": job.status,
                               "error": job.error, "events": job.events}
        if job.status == "done" and job.result and job.notebook:
            nb = job.notebook
            out["notebook"] = {
                "name": Path(job.path).name,
                "language": nb.language,
                "lines": list(nb.lines),
                "cells": [{"number": c.number, "kind": str(c.kind), "start_line": c.start_line,
                           "end_line": c.end_line, "title": c.title, "commented_out": c.commented_out,
                           "magic": c.magic}
                          for c in nb.cells],
            }
            out["claims"] = job.result.claims.model_dump()
            out["documents"] = {k: d.model_dump() for k, d in job.result.documents.items()}
        return out

    @app.get("/api/runs/{run_id}/{audience}.md", response_class=PlainTextResponse)
    def markdown(run_id: str, audience: str) -> str:
        """One document as Markdown with plain-text citations."""
        job = get_job(run_id)
        if job.status != "done" or not job.result or audience not in job.result.documents:
            raise HTTPException(404, "No such document")
        return render_markdown(job.result.documents[audience])

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(STATIC / "index.html")

    app.mount("/static", StaticFiles(directory=STATIC), name="static")
    return app


def main() -> None:
    """Start the app (127.0.0.1:8000 by default) and open the browser after a second.

    The startup line prints the LLM endpoint, so avoid sharing the terminal when the
    endpoint is private.
    """
    settings = get_settings()
    url = f"http://{settings.host}:{settings.port}/"
    print(f"nb-expl running at {url}  (models: extract={settings.model_extract}, "
          f"write={settings.model_write}, endpoint={settings.openai_base_url})")
    if settings.open_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    uvicorn.run(create_app(settings), host=settings.host, port=settings.port, log_level="warning")
