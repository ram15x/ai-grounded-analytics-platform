"""Cassettes: record and replay model calls (PHASE_1C.md section d).

- Only responses to public-tagged requests are recorded. Headers and keys are never stored: an
  entry is built from the request messages and the response text only, and a scan refuses any
  line that looks like a credential.
- The key is a SHA-256 over provider, model ID, the normalised request, the prompt-template hash,
  the semantic-layer hash, the claim-schema hash and a warehouse content hash, so changing any of
  them turns a recorded entry into a replay miss.
- In replay mode a miss raises CassetteMiss. There is no network fallback.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..semantic import default_path
from .types import PUBLIC, CassetteLeak, CassetteMiss, LLMRequest, LLMResponse

KEY_ENV_VARS = ("GROQ_API_KEY", "GEMINI_API_KEY", "OPENROUTER_API_KEY")
LEAK_PATTERNS = [
    re.compile(r"Bearer\s+\S{8,}", re.I),
    re.compile(r"\bx-goog-api-key\b", re.I),
    re.compile(r'"?authorization"?\s*:', re.I),
    re.compile(r"\bgsk_[A-Za-z0-9]{16,}"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{16,}"),
    re.compile(r"\bAIza[0-9A-Za-z_-]{20,}"),
]
FILE_NAME = "calls.jsonl"


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def layer_hash(path: Path | None = None) -> str:
    text = (path or default_path()).read_text("utf-8").replace("\r\n", "\n")
    return sha256_text(text)


def schema_hash() -> str:
    from ..claims import TemplateAnswer  # noqa: PLC0415
    from .schemas import ComposeOutput, PlanOutput  # noqa: PLC0415

    parts = [m.model_json_schema() for m in (PlanOutput, ComposeOutput, TemplateAnswer)]
    return sha256_text(json.dumps(parts, sort_keys=True))


def warehouse_content_hash(readonly_url: str) -> str:
    """A cheap fingerprint of the loaded data: row count and exact sums over the sales view."""
    from ..guard import execute_guarded  # noqa: PLC0415

    sql = (
        "SELECT COUNT(*) AS n, SUM(s.line_amount) AS amount, SUM(s.quantity) AS qty, "
        "MIN(s.invoice_ts) AS t0, MAX(s.invoice_ts) AS t1, "
        "COUNT(DISTINCT s.invoice_no) AS invoices FROM analytics.sales_lines AS s"
    )
    row = execute_guarded(readonly_url, sql).rows[0]
    return sha256_text(json.dumps([str(v) for v in row]))


@dataclass(frozen=True)
class CassetteContext:
    layer_hash: str
    schema_hash: str
    warehouse_hash: str


def normalise_request(req: LLMRequest) -> dict[str, Any]:
    out = {
        "task": req.task,
        "messages": [[r, c] for r, c in req.messages],
        "json_mode": req.json_mode,
        "temperature": req.temperature,
        "max_tokens": req.max_tokens,
        "schema_name": req.schema_name,
        "prompt_id": req.prompt_id,
    }
    if req.reasoning_effort is not None:  # only when set, so existing recordings keep their keys
        out["reasoning_effort"] = req.reasoning_effort
    return out


def make_key(provider: str, model: str, req: LLMRequest, ctx: CassetteContext) -> str:
    payload = {
        "provider": provider,
        "model": model,
        "request": normalise_request(req),
        "prompt_hash": req.prompt_hash,
        "layer_hash": ctx.layer_hash,
        "schema_hash": ctx.schema_hash,
        "warehouse_hash": ctx.warehouse_hash,
    }
    return sha256_text(json.dumps(payload, sort_keys=True, ensure_ascii=True))


def scan_for_secrets(line: str, env: Mapping[str, str]) -> str | None:
    for name in KEY_ENV_VARS:
        value = env.get(name)
        if value and len(value) >= 8 and value in line:
            return f"value of {name}"
    for pat in LEAK_PATTERNS:
        if pat.search(line):
            return f"pattern {pat.pattern}"
    return None


class Cassette:
    def __init__(
        self,
        directory: Path,
        mode: str,
        ctx: CassetteContext,
        *,
        env: Mapping[str, str] | None = None,
        now: Any = time.time,
    ):
        if mode not in ("record", "replay"):
            raise ValueError("mode must be 'record' or 'replay'")
        self.directory = Path(directory)
        self.mode = mode
        self.ctx = ctx
        self.env = os.environ if env is None else env
        self.now = now
        self.entries: dict[str, dict[str, Any]] = {}
        self.hits: set[str] = set()
        self.misses: dict[str, dict[str, Any]] = {}
        self.skipped_internal = 0
        self._load()

    # -- storage --
    def _file(self) -> Path:
        return self.directory / FILE_NAME

    def _load(self) -> None:
        self.entries = read_entries(self.directory)

    def flush(self) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        lines = [json.dumps(self.entries[k], sort_keys=True) for k in sorted(self.entries)]
        self._file().write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    # -- use --
    def lookup(self, provider: str, model: str, req: LLMRequest) -> LLMResponse | None:
        key = make_key(provider, model, req, self.ctx)
        entry = self.entries.get(key)
        if entry is None:
            self.misses[key] = {
                "provider": provider,
                "model": model,
                "task": req.task,
                "prompt_id": req.prompt_id,
            }
            return None
        self.hits.add(key)
        tok = entry["tokens"]
        return LLMResponse(
            text=entry["response"]["text"],
            provider=provider,
            model=model,
            prompt_tokens=tok["prompt"],
            completion_tokens=tok["completion"],
            latency_ms=entry["latency_ms"],
            finish_reason=entry["response"].get("finish_reason", ""),
            tokens_estimated=tok.get("estimated", False),
            from_cassette=True,
            reasoning_tokens=tok.get("reasoning", 0),
        )

    def miss_error(self, req: LLMRequest, tried: list[str]) -> CassetteMiss:
        return CassetteMiss(
            f"no recorded response for task {req.task!r} (prompt {req.prompt_id or '-'}) from "
            f"{', '.join(tried) or 'any eligible provider'}; replay never calls the network. "
            "Re-record by hand with keys in the environment."
        )

    def record(self, provider: str, model: str, req: LLMRequest, resp: LLMResponse) -> bool:
        """Store one response. Returns False (and stores nothing) unless the request is public."""
        if req.sensitivity != PUBLIC:
            self.skipped_internal += 1
            return False
        key = make_key(provider, model, req, self.ctx)
        entry = {
            "key": key,
            "provider": provider,
            "model": model,
            "task": req.task,
            "sensitivity": PUBLIC,
            "request": normalise_request(req),
            "response": {"text": resp.text, "finish_reason": resp.finish_reason},
            "tokens": {
                "prompt": resp.prompt_tokens,
                "completion": resp.completion_tokens,
                "estimated": resp.tokens_estimated,
                "reasoning": resp.reasoning_tokens,
            },
            "latency_ms": resp.latency_ms,
            "recorded_at": datetime.fromtimestamp(self.now(), UTC).isoformat(timespec="seconds"),
            "hashes": {
                "prompt": req.prompt_hash,
                "layer": self.ctx.layer_hash,
                "schema": self.ctx.schema_hash,
                "warehouse": self.ctx.warehouse_hash,
            },
        }
        found = scan_for_secrets(json.dumps(entry), self.env)
        if found:
            raise CassetteLeak(f"refusing to record: the entry contains a credential ({found})")
        self.entries[key] = entry
        self.flush()
        return True

    def usage(self) -> dict[str, Any]:
        return {"hit_keys": sorted(self.hits), "missed": self.misses}

    def write_usage(self, path: Path) -> None:
        Path(path).write_text(json.dumps(self.usage(), indent=2, sort_keys=True), encoding="utf-8")


def read_entries(directory: Path) -> dict[str, dict[str, Any]]:
    path = Path(directory) / FILE_NAME
    out: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return out
    for n, line in enumerate(path.read_text("utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
            out[entry["key"]] = entry
        except (ValueError, KeyError) as exc:
            raise ValueError(f"{path}:{n}: not a cassette entry ({exc})") from exc
    return out


def check_cassettes(
    directory: Path,
    current: dict[str, str | None],
    prompt_hashes: dict[str, str],
    *,
    usage: dict[str, Any] | None = None,
    max_age_days: float | None = None,
    now: float | None = None,
) -> dict[str, Any]:
    """Report problems in a cassette directory.

    current: {"layer": hash, "schema": hash, "warehouse": hash or None (not checked)}.
    stale: entries whose recorded hashes differ from the current ones, so replay would miss them.
    unused / missing: only when a usage file from a replay run is given.
    """
    now = time.time() if now is None else now
    entries = read_entries(directory)
    stale: list[dict[str, Any]] = []
    ages: list[float] = []
    too_old: list[str] = []
    for key, e in entries.items():
        reasons = []
        h = e.get("hashes", {})
        for part in ("layer", "schema", "warehouse"):
            cur = current.get(part)
            if cur is not None and h.get(part) != cur:
                reasons.append(part)
        pid = e.get("request", {}).get("prompt_id", "")
        if pid and prompt_hashes.get(pid) != h.get("prompt"):
            reasons.append("prompt")
        if reasons:
            stale.append({"key": key, "task": e.get("task"), "changed": reasons})
        try:
            age = (now - datetime.fromisoformat(e["recorded_at"]).timestamp()) / 86400
        except (KeyError, ValueError):
            age = float("inf")
        ages.append(age)
        if max_age_days is not None and age > max_age_days:
            too_old.append(key)
    result: dict[str, Any] = {
        "entries": len(entries),
        "stale": stale,
        "too_old": too_old,
        "oldest_age_days": max(ages) if ages else None,
        "newest_age_days": min(ages) if ages else None,
        "warehouse_checked": current.get("warehouse") is not None,
        "unused": None,
        "missing": None,
    }
    if usage is not None:
        hit = set(usage.get("hit_keys", []))
        result["unused"] = sorted(set(entries) - hit)
        result["missing"] = [
            {"key": k, **v} for k, v in sorted(usage.get("missed", {}).items()) if k not in entries
        ]
    result["ok"] = not (
        stale or too_old or result["unused"] or result["missing"]
    )  # None counts as no problem
    return result


def render_check(result: dict[str, Any]) -> str:
    lines = [f"entries: {result['entries']}"]
    if result["oldest_age_days"] is not None:
        lines.append(
            f"age: oldest {result['oldest_age_days']:.1f} days, "
            f"newest {result['newest_age_days']:.1f} days"
        )
    if not result["warehouse_checked"]:
        lines.append("warehouse hash: not checked (no --warehouse-hash and no database URL)")
    for s in result["stale"]:
        lines.append(f"STALE {s['key'][:12]} ({s['task']}): changed {', '.join(s['changed'])}")
    for k in result["too_old"]:
        lines.append(f"TOO OLD {k[:12]}")
    if result["unused"] is None:
        lines.append("unused/missing: not checked (pass --usage from a replay run)")
    else:
        lines += [f"UNUSED {k[:12]}" for k in result["unused"]]
        lines += [f"MISSING {m['key'][:12]} ({m.get('task')})" for m in result["missing"]]
    lines.append("OK" if result["ok"] else "PROBLEMS FOUND")
    return "\n".join(lines)
