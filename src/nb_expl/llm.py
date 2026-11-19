"""OpenAI-compatible LLM client returning validated Pydantic objects.

Structured output support varies between providers and proxies, so the client
tries, in order: response_format=json_schema (strict), json_object, and finally
plain prompting with the schema in the instructions. In "auto" mode the first
mode the server accepts is remembered. Every answer is validated with Pydantic;
invalid JSON is sent back once with the error message for repair.

Responses are cached on disk keyed by everything that affects the answer
(model, mode, messages, schema, temperature), so re-running on the same
notebook is free and deterministic.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import threading
from pathlib import Path
from typing import Any, TypeVar

import openai
from pydantic import BaseModel, ValidationError

from .config import Settings, StructuredMode

log = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)
MODES: list[StructuredMode] = ["json_schema", "json_object", "prompt"]
THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$")


class LLMError(RuntimeError):
    pass


class LLMClient:
    def __init__(self, settings: Settings, client: Any | None = None):
        """`client` lets tests inject a fake OpenAI client. With structured_output=auto,
        the first mode the server accepts is remembered for later calls.
        """
        self.settings = settings
        self.client = client or openai.OpenAI(
            base_url=settings.openai_base_url,
            api_key=settings.openai_api_key,
            timeout=settings.request_timeout,
            max_retries=2,
        )
        self._mode: StructuredMode | None = (
            None if settings.structured_output == "auto" else settings.structured_output
        )
        self._lock = threading.Lock()
        self.calls = 0
        self.cache_hits = 0

    def structured(self, messages: list[dict[str, str]], schema: type[T], model: str, name: str) -> T:
        """Ask for a JSON answer matching `schema` and return it validated."""
        modes = [self._mode] if self._mode else MODES
        last_error: Exception | None = None
        for mode in modes:
            try:
                result = self._structured_with_mode(messages, schema, model, name, mode)
            except openai.BadRequestError as e:  # mode not supported by this server
                log.info("structured mode %s rejected: %s", mode, e)
                last_error = e
                continue
            with self._lock:
                self._mode = self._mode or mode
            return result
        raise LLMError(f"no structured output mode worked: {last_error}")

    def _structured_with_mode(
        self, messages: list[dict[str, str]], schema: type[T], model: str, name: str, mode: StructuredMode
    ) -> T:
        """One structured call in a given mode, with up to max_repair_attempts repair rounds.

        In json_object and prompt modes the schema is appended to the last message. Invalid
        JSON is sent back to the model with the error. Only answers that validate are cached.
        """
        msgs = list(messages)
        json_schema = strict_schema(schema)
        if mode != "json_schema":
            msgs[-1] = {
                **msgs[-1],
                "content": msgs[-1]["content"]
                + "\n\nRespond with a single JSON object, and nothing else, that matches this "
                f"JSON Schema:\n{json.dumps(json_schema)}",
            }
        kwargs: dict[str, Any] = {}
        if mode == "json_schema":
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": name, "schema": json_schema, "strict": True},
            }
        elif mode == "json_object":
            kwargs["response_format"] = {"type": "json_object"}

        for attempt in range(self.settings.max_repair_attempts + 1):
            raw, cached = self._complete(msgs, model, mode, kwargs)
            try:
                result = schema.model_validate(extract_json(raw))
            except (ValueError, ValidationError) as e:
                if attempt == self.settings.max_repair_attempts:
                    raise LLMError(f"invalid {name} JSON after {attempt + 1} attempts: {e}") from e
                msgs = msgs + [
                    {"role": "assistant", "content": raw},
                    {"role": "user", "content": f"That JSON is invalid: {_short_error(e)}\n"
                                                "Return the corrected JSON object only."},
                ]
                continue
            if not cached:  # only answers that validated are cached
                self._store(msgs, model, mode, kwargs, raw)
            return result
        raise AssertionError("unreachable")

    def _complete(self, messages: list[dict[str, str]], model: str, mode: str, kwargs: dict) -> tuple[str, bool]:
        """Raw answer text and whether it came from the cache.

        Raises LLMError for a truncated or empty answer instead of returning partial JSON,
        with a hint when hidden reasoning used up the token budget.
        """
        path = self._cache_path(self._key(messages, model, mode, kwargs))
        if path and path.exists():
            self.cache_hits += 1
            return json.loads(path.read_text())["content"], True
        self.calls += 1
        if self.settings.llm_extra_body:
            kwargs = {**kwargs, "extra_body": self.settings.llm_extra_body}
        resp = self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=self.settings.temperature,
            max_tokens=self.settings.max_tokens,
            **kwargs,
        )
        choice = resp.choices[0]
        content = choice.message.content or ""
        if choice.finish_reason == "length":
            reasoning = _reasoning_text(choice.message)
            hint = (" Most of the budget went to hidden reasoning; disable thinking via LLM_EXTRA_BODY."
                    if len(reasoning) > len(content) else " Lower CHUNK_LINES or raise MAX_TOKENS.")
            raise LLMError(f"answer truncated at MAX_TOKENS={self.settings.max_tokens}.{hint}")
        if not content.strip():
            raise LLMError("model returned an empty answer"
                           + (" (only hidden reasoning)" if _reasoning_text(choice.message) else ""))
        return content, False

    def _store(self, messages: list[dict[str, str]], model: str, mode: str, kwargs: dict, content: str) -> None:
        """Cache a validated answer on disk."""
        path = self._cache_path(self._key(messages, model, mode, kwargs))
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"model": model, "mode": mode, "content": content}))

    def _key(self, messages: list[dict[str, str]], model: str, mode: str, kwargs: dict) -> str:
        """Cache key: hash of everything that affects the answer (messages, model, mode,
        request options, extra body, temperature and max tokens).
        """
        blob = json.dumps(
            {"m": messages, "model": model, "mode": mode, "kw": kwargs, "x": self.settings.llm_extra_body,
             "t": self.settings.temperature, "max": self.settings.max_tokens},
            sort_keys=True,
        )
        return hashlib.sha256(blob.encode()).hexdigest()

    def _cache_path(self, key: str) -> Path | None:
        """Path of a cached answer, or None when caching is disabled."""
        if not self.settings.cache_enabled:
            return None
        return Path(self.settings.cache_dir) / "llm" / f"{key}.json"


def _reasoning_text(message: Any) -> str:
    """Hidden reasoning, which vLLM/OpenRouter-style servers return beside the content."""
    extra = getattr(message, "model_extra", None) or {}
    return (getattr(message, "reasoning_content", None) or extra.get("reasoning_content")
            or extra.get("reasoning") or "")


def extract_json(raw: str) -> Any:
    """Parse JSON from a model answer, tolerating think-blocks, fences and chatter."""
    text = THINK_RE.sub("", raw).strip()
    text = FENCE_RE.sub("", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            raise ValueError("no JSON object in answer") from None
        return json.loads(text[start : end + 1])


def strict_schema(model: type[BaseModel]) -> dict[str, Any]:
    """JSON Schema for OpenAI strict mode: closed objects, every property required."""
    schema = model.model_json_schema()

    def fix(node: Any) -> None:
        if isinstance(node, dict):
            node.pop("default", None)
            if node.get("type") == "object" and "properties" in node:
                node["additionalProperties"] = False
                node["required"] = list(node["properties"])
            for v in node.values():
                fix(v)
        elif isinstance(node, list):
            for v in node:
                fix(v)

    fix(schema)
    return schema


def _short_error(e: Exception, limit: int = 1500) -> str:
    """Error text cut to `limit` characters, so a repair prompt stays small."""
    s = str(e)
    return s if len(s) <= limit else s[:limit] + " ..."
