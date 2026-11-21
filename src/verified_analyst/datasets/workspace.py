"""The upload flow behind the "Your data" screens: stage a file, show its profile, check a filled-in
manifest, load, reconcile, and write the layer. Nothing here calls a model.

Everything lives under one root directory. A staged upload is removed once it is loaded. A dataset
whose definitions do not reproduce the person's own figures is left unpublished: no layer file is
written, so it cannot be asked about."""

from __future__ import annotations

import json
import re
import secrets
import shutil
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from . import load as dl
from . import manifest as dm
from . import profile as dp

TOKEN = re.compile(r"^[0-9a-f]{32}$")
EXTS = (".csv", ".tsv", ".xlsx")
MAX_UPLOAD = 25 * 1024 * 1024  # a browser upload; the file-based CLI keeps the 100 MB cap


class WorkspaceError(ValueError):
    pass


def write_outputs(out: Path, man: dm.Manifest, prof: dp.DatasetProfile, layer, info: dict) -> None:
    """The files that make a loaded dataset usable. Shared with `va dataset load`."""
    out.mkdir(parents=True, exist_ok=True)
    (out / "layer.yaml").write_text(
        yaml.safe_dump(json.loads(layer.model_dump_json()), sort_keys=False), encoding="utf-8"
    )
    (out / "visibility.json").write_text(
        json.dumps({"relation": info["relation"], "visibility": man.visibility}), "utf-8"
    )
    (out / "profile.json").write_text(json.dumps(prof.as_dict(), indent=1, default=str), "utf-8")
    (out / "manifest.yaml").write_text(
        yaml.safe_dump(man.model_dump(mode="json", exclude_defaults=True), sort_keys=False), "utf-8"
    )


class Workspace:
    def __init__(self, root: Path, admin_url: str, readonly_role: str | None):
        self.root = Path(root)
        self.admin_url = admin_url
        self.readonly_role = readonly_role

    # -- staging -------------------------------------------------------------------------------
    def _staged_dir(self, token: str) -> Path:
        if not TOKEN.match(token or ""):
            raise WorkspaceError("unknown upload")
        d = self.root / "_staged" / token
        if not d.is_dir():
            raise WorkspaceError("unknown upload (it may have been loaded already)")
        return d

    def stage(self, filename: str, raw: bytes) -> dict[str, Any]:
        ext = Path(filename or "").suffix.lower()
        if ext not in EXTS:
            raise WorkspaceError("upload a .csv, .tsv or .xlsx file")
        if not raw or len(raw) > MAX_UPLOAD:
            raise WorkspaceError(f"the file must be between 1 byte and {MAX_UPLOAD // 2**20} MB")
        token = secrets.token_hex(16)
        d = self.root / "_staged" / token
        d.mkdir(parents=True)
        path = d / f"upload{ext}"
        path.write_bytes(raw)
        try:
            df = dp.read_table(path)
            prof = dp.profile_frame(df, Path(filename).name)
        except dp.ProfileError as exc:
            shutil.rmtree(d, ignore_errors=True)
            raise WorkspaceError(str(exc)) from exc
        stem = re.sub(r"[^a-z0-9]+", "_", Path(filename).stem.lower()).strip("_") or "data"
        if not stem[0].isalpha():
            stem = f"d_{stem}"
        return {
            "token": token,
            "profile": prof.as_dict(),
            "skeleton": dm.skeleton(prof, stem[:40]),
        }

    def _read(self, token: str) -> tuple[Any, dp.DatasetProfile]:
        d = self._staged_dir(token)
        path = next(d.glob("upload.*"))
        df = dp.read_table(path)
        return df, dp.profile_frame(df, path.name)

    # -- checking and loading ------------------------------------------------------------------
    def _manifest(self, data: dict[str, Any]) -> dm.Manifest:
        try:
            return dm.Manifest.model_validate(data)
        except ValidationError as exc:
            msgs = [f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors()]
            raise WorkspaceError("; ".join(msgs[:8])) from exc

    def check(self, token: str, manifest: dict[str, Any]) -> dict[str, Any]:
        """Validate only: nothing is loaded. Returns the problems, or the metrics that would
        be offered for questions."""
        _, prof = self._read(token)
        man = self._manifest(manifest)
        problems = dm.validate(man, prof)
        out: dict[str, Any] = {"problems": problems, "ok": not problems}
        if not problems:
            layer = dm.build_layer(man, prof)
            out["approved"] = sorted(layer.metrics)
            out["not_approved"] = sorted(
                k for k, s in man.metrics.items() if s.status != "approved"
            )
            out["without_known_figure"] = sorted(
                k for k, s in man.metrics.items() if s.status == "approved" and not s.known
            )
        return out

    def load(self, token: str, manifest: dict[str, Any], overwrite: bool = False) -> dict[str, Any]:
        df, prof = self._read(token)
        man = self._manifest(manifest)
        out = self.root / man.name
        if (out / "layer.yaml").exists() and not overwrite:
            raise WorkspaceError(
                f"a dataset named {man.name!r} already exists; choose to replace it"
            )
        try:
            layer = dm.build_layer(man, prof)
            info = dl.load_dataset(df, prof, man, self.admin_url, self.readonly_role)
            rec = dl.reconcile(man, prof, self.admin_url)
        except (dm.ManifestError, dl.LoadError) as exc:
            raise WorkspaceError(str(exc)) from exc
        failed = [r for r in rec if r["ok"] is False]
        if failed:
            # keep an older published version of this name usable only if it was not replaced
            (out / "layer.yaml").unlink(missing_ok=True)
            raise WorkspaceError(
                "not published: the definitions do not reproduce your figures: "
                + "; ".join(
                    f"{r['metric']} expected {r['expected']}, the data gives {r['got']}"
                    for r in failed
                )
            )
        write_outputs(out, man, prof, layer, info)
        shutil.rmtree(self.root / "_staged" / token, ignore_errors=True)
        return {
            "name": man.name,
            "rows": info["rows"],
            "not_loaded": info["not_loaded"],
            "approved": sorted(layer.metrics),
            "reconciliation": rec,
            "visibility": man.visibility,
        }

    # -- published datasets --------------------------------------------------------------------
    def datasets(self) -> list[dict[str, Any]]:
        out = []
        for p in sorted(self.root.glob("*/layer.yaml")):
            try:
                vis = json.loads((p.parent / "visibility.json").read_text("utf-8"))
                man = yaml.safe_load((p.parent / "manifest.yaml").read_text("utf-8"))
            except (OSError, ValueError, yaml.YAMLError):
                continue
            out.append(
                {
                    "name": p.parent.name,
                    "description": man.get("description", ""),
                    "visibility": vis.get("visibility", "internal"),
                    "examples": [e.get("question") for e in man.get("examples", [])][:3],
                    "metrics": sorted(
                        k
                        for k, s in (man.get("metrics") or {}).items()
                        if s.get("status") == "approved"
                    ),
                }  # fmt: skip
            )
        return out

    def layer_path(self, name: str) -> Path:
        if not re.match(r"^[a-z][a-z0-9_]*$", name or ""):
            raise WorkspaceError("unknown dataset")
        p = self.root / name / "layer.yaml"
        if not p.exists():
            raise WorkspaceError("unknown dataset")
        return p
