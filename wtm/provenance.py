"""Provenance stamps for generated artifacts.

Every artifact should record by itself which code, configuration, inputs and
observable produced it. A manifest records:

  source       sha256 over every source file (wtm/, experiments/, release/,
               packaging files), plus git commit and cleanliness when a git
               checkout is available. Without git (a source ZIP), the files
               are checked against release/SOURCE_MANIFEST.json. Missing git
               metadata is never reported as a verified clean tree.
  config       sha256 of experiments/config/frozen_benchmark.yaml
  environment  Python, platform, and versions of the numerical dependencies
  window       the observation window, decided from the record kernels
  extra        caller-supplied: inputs and their hashes, seed policy, solver
               and bootstrap settings, exclusions, artifact checksums

The commit recorded is HEAD at generation time, i.e. the code that ran.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

__all__ = ["observation_window", "manifest", "write_manifest", "sha256_file",
           "source_files", "source_hash", "source_state", "environment"]

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "experiments" / "config" / "frozen_benchmark.yaml"
SOURCE_MANIFEST = ROOT / "release" / "SOURCE_MANIFEST.json"
SOURCE_GLOBS = ("wtm/**/*.py", "experiments/**/*.py", "experiments/config/*.yaml",
                "release/*.py", "release/*.yaml", "pyproject.toml", "requirements.txt",
                "constraints.txt")
DEPENDENCIES = ("numpy", "scipy", "pandas", "highspy", "mpmath", "yaml", "matplotlib")


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def source_files(root: Path = ROOT) -> list[Path]:
    out = set()
    for g in SOURCE_GLOBS:
        out |= {p for p in root.glob(g) if p.is_file() and "__pycache__" not in p.parts}
    return sorted(out)


def source_hashes(root: Path = ROOT) -> dict:
    return {str(p.relative_to(root)): sha256_file(p) for p in source_files(root)}


def source_hash(root: Path = ROOT) -> str:
    """One digest over (path, sha256) of every source file."""
    h = hashlib.sha256()
    for rel, digest in sorted(source_hashes(root).items()):
        h.update(f"{rel}\0{digest}\n".encode())
    return h.hexdigest()


def _git(*args) -> tuple[int, str]:
    try:
        r = subprocess.run(("git", *args), capture_output=True, text=True, cwd=ROOT, timeout=30)
        return r.returncode, r.stdout.rstrip("\n")
    except Exception:
        return 1, ""


def source_state(root: Path = ROOT) -> dict:
    """Which code ran: a git commit and cleanliness, or a release-manifest check."""
    rc, top = _git("rev-parse", "--show-toplevel")
    top = top.strip()
    if rc == 0 and top and Path(top).resolve() == root.resolve():
        _, commit = _git("rev-parse", "HEAD")
        _, branch = _git("rev-parse", "--abbrev-ref", "HEAD")
        # paths are given relative to the repository root (cwd=ROOT); untracked
        # source files count as dirty, since they are code that ran
        _, status = _git("status", "--porcelain", "--untracked-files=all", "--",
                         "wtm", "experiments/scripts", "experiments/tests",
                         "experiments/config", "experiments/audit", "release")
        dirty = [l[3:] for l in status.splitlines() if l.strip()]
        return {"kind": "git", "code_commit": commit, "code_branch": branch,
                "source_dirty": bool(dirty), "dirty_paths": dirty[:50]}
    if SOURCE_MANIFEST.exists():
        want = json.loads(SOURCE_MANIFEST.read_text())["files"]
        have = source_hashes(root)
        changed = sorted(k for k in want if have.get(k) != want[k])
        added = sorted(set(have) - set(want))
        return {"kind": "release-source-manifest",
                "release": json.loads(SOURCE_MANIFEST.read_text()).get("release"),
                "matches_manifest": not changed and not added,
                "changed": changed[:50], "added": added[:50]}
    return {"kind": "unverified", "note": "no git metadata and no release/SOURCE_MANIFEST.json"}


def environment() -> dict:
    vers = {}
    for name in DEPENDENCIES:
        try:
            mod = __import__(name)
            vers[name] = getattr(mod, "__version__", "unknown")
        except Exception:
            vers[name] = "absent"
    return {"python": sys.version.split()[0], "implementation": platform.python_implementation(),
            "platform": platform.platform(), "machine": platform.machine(), "packages": vers}


def observation_window(prefixes=(1, 2, 3, 4)) -> str:
    """'A', 'B', or 'unknown', decided from the record kernels themselves.

    Window A (the paper's): kernels of degree <= 2t-1, exact rank 2t.
    Window B (superseded):  degree 2t, rank 2t+1 -- it lets the planner see
    who fails at date t, which is only visible at t+1.
    """
    from .npmle import record_matrix
    from .observation import individual_functionals

    grid = np.linspace(0.0, 1.0, 200)
    verdicts = set()
    for t in prefixes:
        deg = max(f.degree for _, f in individual_functionals(t, 1))
        rank = int(np.linalg.matrix_rank(record_matrix(grid, t, 1)))
        if deg <= 2 * t - 1 and rank == 2 * t:
            verdicts.add("A")
        elif deg <= 2 * t and rank == 2 * t + 1:
            verdicts.add("B")
        else:
            verdicts.add("unknown")
    return verdicts.pop() if len(verdicts) == 1 else "unknown"


def manifest(inputs=(), artifacts=(), **extra) -> dict:
    """Provenance record for an artifact generated by the current tree.

    `inputs` and `artifacts` are paths; each is recorded with its sha256.
    """
    from .observation import individual_functionals

    def rel(p):
        p = Path(p).resolve()
        try:
            return str(p.relative_to(ROOT))
        except ValueError:
            return str(p)

    return {
        "source": {"sha256": source_hash(), **source_state()},
        "config": {"path": rel(CONFIG), "sha256": sha256_file(CONFIG)},
        "environment": environment(),
        "observation_window": observation_window(),
        "record_atom_count_tau4": len(individual_functionals(4, 1)),
        "inputs": {rel(p): sha256_file(p) for p in inputs},
        "artifacts": {rel(p): sha256_file(p) for p in artifacts},
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        **extra,
    }


def write_manifest(path, **extra) -> dict:
    m = manifest(**extra)
    Path(path).write_text(json.dumps(m, indent=2, sort_keys=True, default=str) + "\n")
    return m
