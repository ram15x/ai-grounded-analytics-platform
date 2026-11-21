"""Reports: a draft collects answers, a person approves it, and only then is a file written.

Nothing here sends anything. Only answers whose outcome is "verified" enter the body; every other
question is listed under "Withheld" with its reason, so a reader sees what was left out.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PENDING, APPROVED = "pending_approval", "approved"


class ReportError(ValueError):
    pass


def draft(title: str, runs: list[dict[str, Any]], now: datetime | None = None) -> dict[str, Any]:
    """Build a pending report from QuestionRun.as_dict() records."""
    if not title.strip():
        raise ReportError("a report needs a title")
    stamp = (now or datetime.now(UTC)).strftime("%Y-%m-%dT%H:%M:%SZ")
    body = [r for r in runs if r["outcome"] == "verified"]
    left_out = [r for r in runs if r["outcome"] != "verified"]
    rep = {
        "title": title.strip(),
        "created": stamp,
        "status": PENDING,
        "sections": [
            {
                "id": r.get("id", ""),
                "question": r["question"],
                "text": r["rendered"],
                "numbers_verified": len(_verdicts(r)),
                "sql": (r.get("evidence") or {}).get("sql", ""),
                "passages": [
                    p["id"] for p in ((r.get("evidence") or {}).get("passages") or {}).values()
                ],
            }
            for r in body
        ],
        "withheld": [
            {"id": r.get("id", ""), "question": r["question"], "reason": r.get("reason", "")}
            for r in left_out
        ],
        "approved_by": None,
        "approved_at": None,
    }
    rep["digest"] = _digest(rep)
    return rep


def _verdicts(r: dict[str, Any]) -> list[dict[str, Any]]:
    return ((r.get("verification") or {}).get("verification") or {}).get("verdicts", [])


def _digest(rep: dict[str, Any]) -> str:
    core = {k: rep[k] for k in ("title", "sections", "withheld")}
    return hashlib.sha256(json.dumps(core, sort_keys=True).encode()).hexdigest()[:16]


def approve(
    rep: dict[str, Any], who: str, digest: str, now: datetime | None = None
) -> dict[str, Any]:
    """Approval names the content: it must quote the digest of the draft the approver saw."""
    if rep["status"] != PENDING:
        raise ReportError("this report is not pending approval")
    if not who.strip():
        raise ReportError("approval needs a name")
    if digest != _digest(rep) or digest != rep["digest"]:
        raise ReportError("digest does not match the draft; the report changed after it was read")
    if not rep["sections"]:
        raise ReportError("nothing verified to publish")
    out = dict(rep)
    out["status"], out["approved_by"] = APPROVED, who.strip()
    out["approved_at"] = (now or datetime.now(UTC)).strftime("%Y-%m-%dT%H:%M:%SZ")
    return out


def to_markdown(rep: dict[str, Any]) -> str:
    lines = [f"# {rep['title']}", "", f"Status: {rep['status']}  ·  Draft digest: {rep['digest']}"]
    if rep["approved_by"]:
        lines.append(f"Approved by {rep['approved_by']} at {rep['approved_at']}")
    lines.append("")
    for s in rep["sections"]:
        lines += [
            f"## {s['question']}",
            "",
            s["text"],
            "",
            f"_{s['numbers_verified']} number(s) verified (warehouse or stored passage)._",
            "",
        ]
        if s["sql"]:
            lines += ["```sql", s["sql"], "```", ""]
        if s["passages"]:
            lines += ["Sources: " + ", ".join(s["passages"]), ""]
    if rep["withheld"]:
        lines += ["## Withheld", ""]
        lines += [f"- {w['question']} ({w['reason'] or 'not verified'})" for w in rep["withheld"]]
    return "\n".join(lines) + "\n"


def publish(rep: dict[str, Any], out_dir: Path) -> Path:
    """Write the approved report to a file. Refuses a report that is not approved."""
    if rep["status"] != APPROVED:
        raise ReportError("only an approved report is written")
    out_dir.mkdir(parents=True, exist_ok=True)
    name = "".join(c if c.isalnum() else "-" for c in rep["title"].lower()).strip("-")
    path = out_dir / f"{name}-{rep['digest']}.md"
    path.write_text(to_markdown(rep), encoding="utf-8")
    return path
