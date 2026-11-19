import logging

from nb_expl.utils import chat_messages, fill, parallel_map, reporter


def test_parallel_map_keeps_order():
    assert parallel_map(lambda x: x * 2, [3, 1, 2], concurrency=3) == [6, 2, 4]


def test_parallel_map_treats_zero_concurrency_as_one():
    assert parallel_map(str, [1, 2], concurrency=0) == ["1", "2"]


def test_chat_messages():
    assert chat_messages("sys", "hi") == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hi"},
    ]


def test_fill_replaces_placeholders():
    assert fill("{{a}} and {{b}}", a=1, b="x") == "1 and x"


def test_reporter_falls_back_to_logger(caplog):
    logger = logging.getLogger("test_utils")
    with caplog.at_level(logging.INFO, logger="test_utils"):
        reporter(None, logger)("hello")
    assert "hello" in caplog.text
    seen = []
    reporter(seen.append, logger)("x")
    assert seen == ["x"]
