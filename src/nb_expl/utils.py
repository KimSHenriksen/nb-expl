"""Small helpers shared across modules."""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from importlib import resources
from typing import TypeVar

T = TypeVar("T")
R = TypeVar("R")

Progress = Callable[[str], None]


def reporter(progress: Progress | None, logger: logging.Logger) -> Progress:
    """Progress callback for a pipeline step.

    Each step reports one human-readable line per stage (the web app streams them
    over SSE, the CLIs print them). Without a callback, the lines go to the calling
    module's logger at INFO level.
    """
    return progress or logger.info


def load_prompt(name: str) -> str:
    """Read a prompt template from the `prompts/` package directory."""
    return resources.files("nb_expl.prompts").joinpath(f"{name}.md").read_text(encoding="utf-8")


def fill(template: str, **values: object) -> str:
    """Replace each `{{key}}` placeholder in `template` with `str(value)`."""
    for k, v in values.items():
        template = template.replace("{{" + k + "}}", str(v))
    return template


def chat_messages(system: str, user: str) -> list[dict[str, str]]:
    """A system + user message pair in the OpenAI chat format."""
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def parallel_map(fn: Callable[[T], R], items: Iterable[T], concurrency: int) -> list[R]:
    """Apply `fn` to each item with up to `concurrency` threads; results keep input order."""
    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        return list(pool.map(fn, items))
