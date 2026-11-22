"""`grounded docs binder-locked`: score the pre-registered binder variants on the locked half, once.

This is the only module that builds the locked half of the binder study. It accepts no settings:
the variants and the decision rule are fixed in docs/adr/0011-binder-variants.md and in
`PRE_REGISTERED` below, and the run refuses unless the ADR is committed, unchanged since, and
agrees with the code. The ledger allows one run. It never changes the shipped default; the owner
decides after reading the table.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import yaml

from .binder_study import CORPUS_DIR, build_cases, evaluate, load_numbers
from .binding import DEFAULT, BinderConfig
from .store import DocumentError

Git = Callable[[Sequence[str]], tuple[int, str]]
ROOT = Path(__file__).resolve().parents[3]
ADR_PATH = ROOT / "docs" / "adr" / "0011-binder-variants.md"
LEDGER_PATH = ROOT / "data" / "binder_locked_ledger.jsonl"
# The locked half was scored once (2026-11-21) and informed the decision in ADR-0012. It cannot
# be scored again, and nothing else may use it to choose a rule.
SPENT_BY: str | None = "docs/adr/0012-binder-default.md"


def _spec(frac: float, window: int, skip: bool, scope: str = "window", nearest: bool = False):
    return {
        "frac": frac,
        "window": window,
        "ignore_repeated": False,
        "head_noun": False,
        "skip_units": skip,
        "scope": scope,
        "nearest": nearest,
    }


PRE_REGISTERED: dict[str, dict[str, Any]] = {
    "current": _spec(1.0, 6, False),
    "C1": _spec(1.0, 6, True),
    "C2": _spec(0.75, 6, True),
    "S1": _spec(1.0, 6, True, "sentence"),
    "S2": _spec(1.0, 8, True, "sentence", True),
    "SL": _spec(1.0, 6, False, "sentence"),
}
RULE = (
    "A candidate is put forward for review only if, on the locked half, its false accepts on the "
    "swapped cases are no higher than current's AND its false rejects on the correct cases are "
    "lower than current's. Otherwise the shipped rule (current) stays. The absolute target of 0 "
    "false accepts is not expected to be met by any registered candidate and is reported as it "
    "falls."
)
NOTE = (
    "Synthetic corpus written by the project author, generated labels, counts not a benchmark. "
    "Nothing here is an accuracy claim. The owner decides; this run changes no default."
)


def _git(root: Path) -> Git:
    def run(args: Sequence[str]) -> tuple[int, str]:
        try:
            p = subprocess.run(  # noqa: S603
                ["git", "-C", str(root), *args],  # noqa: S607
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            return 127, str(exc)
        return p.returncode, p.stdout.strip()

    return run


def parse_candidates(text: str) -> dict[str, Any]:
    start, end = "<!-- candidates:begin -->", "<!-- candidates:end -->"
    if start not in text or end not in text:
        raise DocumentError("the ADR has no machine-readable candidates block")
    body = text.split(start, 1)[1].split(end, 1)[0].replace("```yaml", "").replace("```", "")
    try:
        data = yaml.safe_load(body)
    except yaml.YAMLError as exc:
        raise DocumentError("the ADR candidates block is not valid YAML") from exc
    if not isinstance(data, dict) or not isinstance(data.get("candidates"), dict):
        raise DocumentError("the ADR candidates block has no `candidates:` mapping")
    return data["candidates"]


def config_of(name: str) -> BinderConfig:
    return BinderConfig(**PRE_REGISTERED[name])


def check_adr(adr_path: Path = ADR_PATH, root: Path = ROOT, git: Git | None = None) -> str:
    """Refuse unless the ADR is committed, unchanged since, and matches the code. Returns its
    sha256."""
    git = git or _git(root)
    rel = adr_path.relative_to(root).as_posix() if adr_path.is_absolute() else str(adr_path)
    if not adr_path.is_file():
        raise DocumentError(f"refusing to run: the ADR {rel} is missing")
    code, out = git(["rev-parse", "--is-inside-work-tree"])
    if code != 0 or out != "true":
        raise DocumentError(
            "refusing to run: cannot confirm the ADR is committed (no git checkout)"
        )
    if git(["ls-files", "--error-unmatch", "--", rel])[0] != 0:
        raise DocumentError(f"refusing to run: the ADR {rel} has not been committed")
    code, out = git(["status", "--porcelain", "--", rel])
    if code != 0 or out:
        raise DocumentError(f"refusing to run: the ADR {rel} has changed since it was committed")
    text = adr_path.read_text("utf-8")
    if parse_candidates(text) != PRE_REGISTERED:
        raise DocumentError(
            "refusing to run: the variants in the ADR do not match the ones built into the code"
        )
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _entries(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(x) for x in path.read_text("utf-8").splitlines() if x.strip()]


def decide(rows: dict[str, dict[str, Any]]) -> dict[str, Any]:
    cur = rows["current"]
    out: dict[str, Any] = {}
    for name, r in rows.items():
        if name == "current":
            continue
        fa_ok = r["false_accept"]["k"] <= cur["false_accept"]["k"]
        fr_ok = r["false_reject"]["k"] < cur["false_reject"]["k"]
        out[name] = {"false_accepts_not_higher": fa_ok, "false_rejects_lower": fr_ok,
                     "put_forward": fa_ok and fr_ok}  # fmt: skip
    return out


def run_locked(
    *,
    candidates_arg: str,
    adr_path: Path = ADR_PATH,
    root: Path = ROOT,
    git: Git | None = None,
    ledger: Path = LEDGER_PATH,
    corpus_dir: Path = CORPUS_DIR,
) -> dict[str, Any]:
    given = sorted(c.strip() for c in candidates_arg.split(","))
    if given != sorted(PRE_REGISTERED):
        raise DocumentError(
            "only the pre-registered variants can be scored, all together: "
            "--candidates " + ",".join(PRE_REGISTERED)
        )
    if SPENT_BY is not None:
        raise DocumentError(f"refusing to run: the locked half is spent ({SPENT_BY})")
    sha = check_adr(adr_path, root, git)
    if _entries(ledger):
        raise DocumentError("refusing to run: the ledger shows the locked half was already scored")
    cases = build_cases(load_numbers("locked", corpus_dir))
    rows = {n: evaluate(config_of(n), cases) for n in PRE_REGISTERED}
    report = {
        "half": "locked",
        "documents": len({c.number.doc_key for c in cases}),
        "adr_sha256": sha,
        "shipped_default": DEFAULT.name,
        "rule": RULE,
        "note": NOTE,
        "rows": rows,
        "decision": decide(rows),
    }
    ledger.parent.mkdir(parents=True, exist_ok=True)
    with ledger.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"at": time.strftime("%Y-%m-%dT%H:%M:%S"), "adr_sha256": sha}) + "\n")
    return report


def render_report(rep: dict[str, Any]) -> str:
    lines = [
        f"Binder variants, locked half: {rep['documents']} documents (scored once)",
        rep["note"],
        "",
        f"Rule: {rep['rule']}",
        "",
        f"{'variant':10} {'false reject (correct)':>26} {'false accept (swapped)':>26}  decision",
    ]
    for n, r in rep["rows"].items():
        fr, fa = r["false_reject"], r["false_accept"]
        d = rep["decision"].get(n)
        verdict = "-" if d is None else ("put forward" if d["put_forward"] else "keep current")
        lines.append(
            f"{n:10} {fr['k']:>5} of {fr['n']:<5} (n={fr['n']}) "
            f"{fa['k']:>5} of {fa['n']:<5} (n={fa['n']})  {verdict}"
        )
    return "\n".join(lines) + "\n"


__all__: Sequence[str] = ["PRE_REGISTERED", "run_locked", "render_report"]
