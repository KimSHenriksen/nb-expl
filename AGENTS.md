# AGENTS.md

Instructions for AI coding agents working in this repository. Humans should start with [README.md](README.md).

## What this is

A local web app that reads a Databricks notebook (source export, usually `.py`) and generates three documents: for a manager, a new developer and an AI agent. Every factual sentence must cite `C<cell>:L<line>` in the notebook. The core is deterministic; the LLM only works inside checked rails.

Input is a Databricks source export. Python (`.py`) is the main case and the only language with full static analysis (AST). SQL (`.sql`), Scala (`.scala`) and R (`.r`) exports are parsed too; SQL and Scala cells only get table reads and writes found by regular expressions, and R cells rely on the model alone. Keep new features working for `.py` first.

## Commands

```bash
uv run nb-expl                           # start the app on http://127.0.0.1:8000
uv run pytest                            # all tests; no network, no LLM
uv run python -m nb_expl.pipeline examples/cash_application_engine.py --out examples/output
uv run python -m nb_expl.claims examples/cash_application_engine.py /tmp/claims.json
```

Run `uv run pytest` before you consider a change done.

## Security

- Never print, log, return or commit `OPENAI_API_KEY` or the contents of `.env`. `/api/config` must not expose it.
- The server reads arbitrary local paths, so it binds to `127.0.0.1` only. Do not change the default host.
- No private URLs, hostnames or personal details in committed files.

## Tests

- Tests use `tests/fakes.py` (`fake_client`) instead of a real endpoint. Never add a test that needs network or a running LLM.
- New behaviour in parser/analysis/claims/documents gets a test against `examples/cash_application_engine.py` or a small inline notebook.

## Expensive things — ask first

- Do not clear `.cache/` or regenerate `examples/output/` without being asked.
- Prompt changes invalidate the cache for that step.

## Git workflow

- One feature branch per piece of work (`feature/<name>`). Commit there.
- Do not commit the following: `.env`, `.cache/`.

## Style

- Python 3.11+, type hints, `from __future__ import annotations`, Pydantic models for anything crossing the LLM boundary.
- Code, comments, docs and commit messages in English.
- Keep a library file for shared utility functions.
- When you change code, update the comments and docstrings that describe it. Do not delete comments that are still true.
- Comment why, not what: the reason for a rule, a workaround or a non-obvious choice.
- Give every function and method a short docstring for a human reviewer: what it does, what it returns, and anything surprising (side effects, error handling, assumptions). One or two lines is usually enough.
