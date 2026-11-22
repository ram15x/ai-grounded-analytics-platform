"""Every database connection gets a connect timeout (default 10 s, GA_CONNECT_TIMEOUT_S).

No real connection is opened here: `psycopg.connect` is replaced by a recorder that fails, so the
tests are fast and need no database."""

from pathlib import Path

import psycopg
import pytest

from grounded_analytics import guard
from grounded_analytics.config import (
    DEFAULT_CONNECT_TIMEOUT_S,
    ConfigError,
    connect_kwargs,
    connect_timeout_s,
)
from grounded_analytics.warehouse import load


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("GA_CONNECT_TIMEOUT_S", raising=False)


def test_default_is_ten_seconds():
    assert DEFAULT_CONNECT_TIMEOUT_S == 10
    assert connect_timeout_s() == 10
    assert connect_kwargs("host=localhost") == {"connect_timeout": 10}


@pytest.mark.parametrize("raw, want", [("1", 1), (" 25 ", 25), ("300", 300)])
def test_env_sets_the_timeout(monkeypatch, raw, want):
    monkeypatch.setenv("GA_CONNECT_TIMEOUT_S", raw)
    assert connect_kwargs("host=localhost") == {"connect_timeout": want}


@pytest.mark.parametrize("raw", ["0", "-5", "301", "ten", "1.5", "1e2"])
def test_bad_values_are_a_config_error_not_a_silent_default(monkeypatch, raw):
    monkeypatch.setenv("GA_CONNECT_TIMEOUT_S", raw)
    with pytest.raises(ConfigError, match="GA_CONNECT_TIMEOUT_S"):
        connect_timeout_s()


def test_empty_value_means_default(monkeypatch):
    monkeypatch.setenv("GA_CONNECT_TIMEOUT_S", "  ")
    assert connect_timeout_s() == 10


@pytest.mark.parametrize(
    "url", ["host=localhost connect_timeout=3", "postgresql://u@h/d?connect_timeout=3"]
)
def test_a_timeout_already_in_the_url_is_kept(url):
    assert connect_kwargs(url) == {}


class _Stop(Exception):
    pass


def _record(monkeypatch, calls):
    def fake(conninfo="", **kwargs):
        calls.append((conninfo, kwargs))
        raise _Stop

    monkeypatch.setattr(psycopg, "connect", fake)


def test_guarded_queries_pass_the_timeout(monkeypatch):
    calls = []
    _record(monkeypatch, calls)
    monkeypatch.setenv("GA_CONNECT_TIMEOUT_S", "4")
    with pytest.raises(_Stop):
        guard.execute_guarded("host=localhost", "SELECT 1 AS x FROM analytics.products")
    assert calls[0][1]["connect_timeout"] == 4


def test_the_loader_passes_the_timeout(monkeypatch, tmp_path):
    calls = []
    _record(monkeypatch, calls)
    monkeypatch.setattr(load, "read_raw", lambda path: None)
    monkeypatch.setattr(load, "normalize", lambda raw: None)
    with pytest.raises(_Stop):
        load.load_file(Path(tmp_path / "x.csv"), "host=localhost")
    assert calls[0][1]["connect_timeout"] == 10
    assert calls[0][1]["autocommit"] is False


def test_the_cli_reports_a_bad_timeout_value_as_a_clean_error(monkeypatch, capsys, tmp_path):
    from grounded_analytics.cli import main

    answer = tmp_path / "a.json"
    answer.write_text('{"prose": "x"}')
    monkeypatch.setenv("GA_READONLY_DATABASE_URL", "host=localhost")
    monkeypatch.setenv("GA_CONNECT_TIMEOUT_S", "soon")
    monkeypatch.setattr(psycopg, "connect", lambda *a, **k: pytest.fail("connected"))
    assert main(["verify", str(answer)]) == 2
    assert "GA_CONNECT_TIMEOUT_S" in capsys.readouterr().err
