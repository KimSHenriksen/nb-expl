"""End-to-end run: notebook file -> facts -> validated claims -> three documents.

Used by the web app, and from the command line:

    uv run python -m nb_expl.pipeline examples/cash_application_engine.py --out examples/output
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

from pydantic import BaseModel

from .analysis import analyze
from .claims import ClaimExtractor, ClaimSet
from .config import Settings, get_settings
from .documents import DOCUMENTS, Document, DocumentWriter, github_line_link, render_markdown
from .llm import LLMClient
from .parser import Notebook, parse_file
from .utils import Progress, reporter

log = logging.getLogger(__name__)


class RunResult(BaseModel):
    notebook_path: str
    claims: ClaimSet
    documents: dict[str, Document]


def run(path: str | Path, settings: Settings | None = None, progress: Progress | None = None,
        llm: LLMClient | None = None) -> tuple[Notebook, RunResult]:
    """Parse, analyse, extract and verify claims, and write the three documents.

    `progress` receives one line per step. Returns the parsed notebook (for showing
    code) and the run result.
    """
    settings = settings or get_settings()
    say = reporter(progress, log)
    llm = llm or LLMClient(settings)
    say(f"parsing {path}")
    nb = parse_file(path)
    say(f"{len(nb.cells)} cells, {nb.line_count} lines")
    sheet = analyze(nb)
    claims = ClaimExtractor(settings, llm).run(nb, sheet=sheet, progress=say)
    writer = DocumentWriter(settings, llm)
    name = Path(path).name
    docs = {aud: writer.write(aud, claims, name, progress=say) for aud in DOCUMENTS}
    say("done")
    return nb, RunResult(notebook_path=str(path), claims=claims, documents=docs)


def save(result: RunResult, out_dir: str | Path) -> list[Path]:
    """Write the three documents as Markdown (citations link to the notebook's lines) plus JSON."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rel = os.path.relpath(Path(result.notebook_path).resolve(), out.resolve())
    link = github_line_link(rel)
    written = []
    for aud, doc in result.documents.items():
        p = out / f"{aud}.md"
        p.write_text(render_markdown(doc, link=link), encoding="utf-8")
        written.append(p)
    p = out / "run.json"
    p.write_text(json.dumps(result.model_dump(), indent=1), encoding="utf-8")
    written.append(p)
    return written


def main() -> None:
    """Command-line entry: run the pipeline on one notebook and save the documents."""
    ap = argparse.ArgumentParser(description="Generate cited documentation for a Databricks notebook")
    ap.add_argument("notebook")
    ap.add_argument("--out", default="output", help="output directory (default: output)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(message)s")
    _, result = run(args.notebook, progress=lambda m: print(m, flush=True))
    for p in save(result, args.out):
        print("wrote", p)


if __name__ == "__main__":
    main()
