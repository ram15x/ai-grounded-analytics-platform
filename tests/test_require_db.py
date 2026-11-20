"""VA_REQUIRE_DB=1 turns "database unreachable" from a skip into a failure (used by CI)."""

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent


def run_pytest(env_extra: dict[str, str], drop: tuple[str, ...] = ()):
    env = {k: v for k, v in os.environ.items() if k not in drop and not k.startswith("VA_")}
    env.update(env_extra)
    return subprocess.run(
        [sys.executable, "-m", "pytest", "-o", "addopts=", "tests/test_cli.py"],
        cwd=ROOT, env=env, capture_output=True, text=True, timeout=120, check=False,
    )  # fmt: skip


def test_unreachable_db_skips_by_default():
    r = run_pytest({"VA_TEST_DATABASE_URL": "host=127.0.0.1 port=1 user=x dbname=y"})
    assert r.returncode == 0, r.stdout + r.stderr
    assert "skipped" in r.stdout


def test_unreachable_db_fails_when_required():
    r = run_pytest({"VA_TEST_DATABASE_URL": "host=127.0.0.1 port=1 user=x dbname=y",
                    "VA_REQUIRE_DB": "1"})  # fmt: skip
    assert r.returncode != 0
    out = r.stdout + r.stderr
    assert "VA_REQUIRE_DB=1" in out and "not reachable" in out


def test_missing_url_fails_when_required():
    r = run_pytest({"VA_REQUIRE_DB": "1"})
    assert r.returncode != 0
    assert "VA_TEST_DATABASE_URL" in r.stdout + r.stderr


def test_require_flag_ignored_when_no_db_tests_selected():
    env = {k: v for k, v in os.environ.items() if not k.startswith("VA_")}
    env["VA_REQUIRE_DB"] = "1"
    r = subprocess.run(
        [sys.executable, "-m", "pytest", "-o", "addopts=", "tests/test_package.py"],
        cwd=ROOT, env=env, capture_output=True, text=True, timeout=120, check=False,
    )  # fmt: skip
    assert r.returncode == 0, r.stdout + r.stderr


def test_cli_missing_env_is_one_clean_line(tmp_path):
    env = {k: v for k, v in os.environ.items() if not k.startswith("VA_")}
    f = tmp_path / "a.json"
    f.write_text('{"prose": "x"}')
    r = subprocess.run(
        [sys.executable, "-m", "verified_analyst.cli", "verify", str(f)],
        env=env, capture_output=True, text=True, timeout=60, check=False,
    )  # fmt: skip
    assert r.returncode == 2
    assert "Traceback" not in r.stderr
    assert r.stderr.strip().count("\n") == 0
    assert "VA_READONLY_DATABASE_URL" in r.stderr
