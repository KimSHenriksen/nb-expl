"""Fake OpenAI client for tests: no network, queued or computed answers."""

from types import SimpleNamespace

import openai

from nb_expl.config import Settings
from nb_expl.llm import LLMClient


class FakeCompletions:
    """Returns queued answers; a callable answer receives the request kwargs."""

    def __init__(self, answers):
        self.answers = list(answers)
        self.requests = []

    def create(self, **kwargs):
        self.requests.append(kwargs)
        answer = self.answers.pop(0) if self.answers else '{"claims": []}'
        if callable(answer):
            answer = answer(kwargs)
        if isinstance(answer, Exception):
            raise answer
        finish = "stop"
        if isinstance(answer, tuple):  # (content, finish_reason, reasoning)
            answer, finish, reasoning = answer
            msg = SimpleNamespace(content=answer, reasoning_content=reasoning)
        else:
            msg = SimpleNamespace(content=answer)
        return SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason=finish)])


def fake_client(answers, tmp_path, **overrides):
    completions = FakeCompletions(answers)
    settings = Settings(_env_file=None, cache_dir=tmp_path, concurrency=1, **overrides)
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    return LLMClient(settings, client=client), completions, settings


def bad_request():
    resp = SimpleNamespace(status_code=400, headers={}, request=None)
    return openai.BadRequestError("response_format not supported", response=resp, body=None)
