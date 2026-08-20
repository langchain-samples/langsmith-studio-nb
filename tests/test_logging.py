import logging

from langsmith_studio_nb._logging import NOISY_LOGGERS, silence_loggers


class FakeLogger:
    def __init__(self):
        self.level = logging.INFO

    def setLevel(self, level):  # noqa: N802 - matches logging.Logger
        self.level = level


def test_silence_loggers_raises_every_level():
    loggers = {name: FakeLogger() for name in NOISY_LOGGERS}

    silence_loggers(logging.ERROR, get_logger=loggers.__getitem__)

    assert {logger.level for logger in loggers.values()} == {logging.ERROR}


def test_silence_loggers_can_restore_every_level():
    loggers = {name: FakeLogger() for name in NOISY_LOGGERS}
    restore = silence_loggers(logging.ERROR, get_logger=loggers.__getitem__)

    restore()

    assert {logger.level for logger in loggers.values()} == {logging.INFO}


def test_silence_loggers_covers_the_server_and_its_tunnel():
    assert "langgraph_api" in NOISY_LOGGERS  # also covers langgraph_api.tunneling.*
    assert "langgraph_api.server" in NOISY_LOGGERS
    assert "langgraph_runtime_inmem" in NOISY_LOGGERS
    assert "uvicorn.error" in NOISY_LOGGERS


def test_silence_loggers_applies_before_import():
    """A placeholder logger keeps its level once the real module is imported."""
    name = "langsmith_studio_nb_test_not_yet_imported"

    silence_loggers(logging.ERROR, names=[name])

    assert logging.getLogger(name).level == logging.ERROR
