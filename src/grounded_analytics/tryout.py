"""The public tryout: a fixed menu plus a few free-text slots, with a per-visitor allowance.

The allowance is a courtesy limit that keeps one visitor inside a free-tier quota. It is held in
memory and keyed by a random cookie, so a visitor who clears cookies gets a fresh one: it is not
a security control. The cost of each answer is carried by the visitor's own API key, which is
used for one request and never stored (see web/app.py)."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PATH = ROOT / "config" / "tryout.yaml"
MENU_TIERS = ("easy", "medium", "hard")
TIERS = (*MENU_TIERS, "own")


class TryoutError(ValueError):
    pass


def load_config(path: Path | None = None) -> dict[str, Any]:
    path = path or DEFAULT_PATH
    try:
        cfg = yaml.safe_load(path.read_text("utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise TryoutError(f"cannot load {path}: {exc}") from exc
    per = cfg.get("per_visitor") or {}
    for t in TIERS:
        v = per.get(t)
        if not isinstance(v, int) or isinstance(v, bool) or v < 0:
            raise TryoutError(f"per_visitor.{t} must be a whole number, 0 or more")
    qs = cfg.get("questions") or {}
    for t in MENU_TIERS:
        items = qs.get(t)
        if not items or not all(isinstance(q, str) and q.strip() for q in items):
            raise TryoutError(f"questions.{t} needs at least one question")
        if len(set(items)) != len(items):
            raise TryoutError(f"questions.{t} has a repeated question")
    if not isinstance(cfg.get("own_max_chars"), int) or cfg["own_max_chars"] < 1:
        raise TryoutError("own_max_chars must be a positive whole number")
    if not isinstance(cfg.get("own_slots"), int) or cfg["own_slots"] < 0:
        raise TryoutError("own_slots must be a whole number, 0 or more")
    if not cfg.get("providers"):
        raise TryoutError("providers is empty")
    return cfg


def public_view(cfg: dict[str, Any]) -> dict[str, Any]:
    """What the page needs. Nothing secret lives in the config, but send only what is used."""
    return {
        "per_visitor": cfg["per_visitor"],
        "own_slots": cfg["own_slots"],
        "own_max_chars": cfg["own_max_chars"],
        "questions": {t: cfg["questions"][t] for t in MENU_TIERS},
        "providers": {
            k: {f: v.get(f) for f in ("label", "key_hint", "note", "model")}
            for k, v in cfg["providers"].items()
        },
    }


class Allowance:
    """Counts answered questions per visitor and tier. Thread-safe."""

    def __init__(self, cfg: dict[str, Any]):
        self.cfg = cfg
        self._used: dict[str, dict[str, int]] = {}
        self._lock = threading.Lock()

    def validate(self, tier: str, question: str) -> str | None:
        """None if this question may be asked in this tier, else the reason."""
        if tier not in TIERS:
            return "unknown tier"
        if tier == "own":
            if not question or len(question) > self.cfg["own_max_chars"]:
                return f"your question must be 1 to {self.cfg['own_max_chars']} characters"
            return None
        if question not in self.cfg["questions"][tier]:
            return "that question is not on the menu"
        return None

    def remaining(self, visitor: str) -> dict[str, int]:
        with self._lock:
            used = self._used.get(visitor, {})
            return {t: max(0, self.cfg["per_visitor"][t] - used.get(t, 0)) for t in TIERS}

    def reserve(self, visitor: str, tier: str) -> bool:
        """Take one use of a tier before the question runs. False when none is left."""
        with self._lock:
            used = self._used.setdefault(visitor, {})
            if used.get(tier, 0) >= self.cfg["per_visitor"][tier]:
                return False
            used[tier] = used.get(tier, 0) + 1
            return True

    def refund(self, visitor: str, tier: str) -> None:
        """Give the use back when the question never got an answer from the model (a rate limit,
        a bad key): the visitor should not lose a slot to a failure that was not an answer."""
        with self._lock:
            used = self._used.setdefault(visitor, {})
            used[tier] = max(0, used.get(tier, 0) - 1)
