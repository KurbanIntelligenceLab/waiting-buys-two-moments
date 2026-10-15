"""Build the compact EM analysis-input bundle from runs/em_mechanism/.

Contents (paths inside the archive, relative to the repository/release root):

  results/em_mechanism_snapshots/finite/<cell>.npz       120 cells, G = 200
  results/em_mechanism_snapshots/finite_G400/<cell>.npz  12 refinement cells
  results/em_mechanism_snapshots/finite_G800/<cell>.npz  12 refinement cells
      uniform-start rows only, in seed order: snap_0, snap_30, snap_55, snap_80,
      snap_<cap> (grid weights, float64), npmle_weights, npmle_curves,
      end_curves and end_ll (endpoint welfare curve and log likelihood), cap,
      seeds
  results/em_mechanism_snapshots/population/<task>.npz   14 exact-probability tasks
      rows (family|init) and every saved snapshot (snap_0, snap_30, snap_55,
      snap_80, snap_<cap>), float64
  results/em_mechanism_snapshots/MANIFEST.json   per-file settings, shapes, sha256,
                                                  sha256 of the source runs/ file
  results/em_mechanism_snapshots/CHECKSUMS.sha256
  results/em_mechanism_snapshots/README.md

Every bundled array is checked to be bitwise identical to the corresponding
array in runs/ (via experiments.em_mechanism.snapshots, which reads either).

Usage: python -m experiments.em_mechanism.build_snapshot_bundle [--out dist]
Writes <out>/wbtm-supplement-2026.10-em-snapshots.zip
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import sys
import zipfile
from pathlib import Path

import numpy as np

from experiments.em_mechanism.common import OUT, ROOT, RUNS

NAME = "wbtm-supplement-2026.10-em-snapshots"
PREFIX = "results/em_mechanism_snapshots"

README = """# EM study: analysis-input snapshot bundle

Full-precision grid weights needed to rerun the EM mechanism analyses
(interventions, contrast attribution, switch points), the maximizer/witness
analysis, the identified-interval positions and the late-EM validation (V7)
from the completed runs, without regenerating the study (~30 core-hours).

## Use

Extract into the root of the repository or of the extracted release, so that
this file ends up at `results/em_mechanism_snapshots/README.md`:

    unzip wbtm-supplement-2026.10-em-snapshots.zip -d <root>
    cd <root> && sha256sum -c results/em_mechanism_snapshots/CHECKSUMS.sha256 --quiet  # run from <root>

`experiments.em_mechanism.snapshots` reads these files automatically when
`runs/em_mechanism/` is absent (force with `WBTM_EM_SNAPSHOTS=bundle`).
The EM study data archive (`wbtm-supplement-2026.10-em-data.zip`) must also be
extracted: the analyses read its saved tables.

## Analysis reproduction (from this bundle)

    export OMP_NUM_THREADS=1
    python -m experiments.em_mechanism.mechanism --workers 4 --out-dir runs/mechanism_check
    python -m experiments.em_mechanism.mechanism_posthoc --workers 4 --out-dir runs/mechanism_check
    python -m experiments.em_mechanism.compare_mechanism_outputs runs/mechanism_check --v7 --maximizers
    python -m experiments.em_mechanism.maximizers --finite finite
    python -m experiments.em_mechanism.validate --skip-v1

`mechanism` re-runs EM only from the bundled EM55 weights to locate exact
switch points (~18 minutes on 4 cores).  `compare_mechanism_outputs` checks
the regenerated tables against the committed ones.

## Not included

Per-iteration records for every initialization (welfare curves, fitted cell
probabilities, moments at all recorded iterations) and the snapshots of the
seven non-uniform initializations in finite samples.  Those are needed only
to regenerate the study itself:

    python -m experiments.em_mechanism.run_population --workers 4   # ~1 core-hour
    python -m experiments.em_mechanism.run_finite --workers 4       # ~30 core-hours

## Manifest

`MANIFEST.json` lists every file with its configuration (from the
generator's checkpoint record), array shapes, sha256, and the sha256 of the
runs/ file it was extracted from.
"""


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def npz_bytes(arrays: dict) -> bytes:
    buf = io.BytesIO()
    np.savez_compressed(buf, **arrays)
    return buf.getvalue()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "dist"))
    a = ap.parse_args(argv)
    os.environ["WBTM_EM_SNAPSHOTS"] = "runs"
    from experiments.em_mechanism import snapshots as S

    members, manifest = {}, {"bundle": NAME, "files": {}, "settings": {
        "finite": "uniform-start rows, seed order; seeds 1001..1050 (G=200) or 1001..1020 (refinement)",
        "population": "all 40 rows (5 families x 8 initializations) per task"}}
    for sub in ("finite", "finite_G400", "finite_G800"):
        for done in sorted((OUT / sub / "cells").glob("*.done.json")):
            key = done.name.replace(".done.json", "")
            meta = json.loads(done.read_text())["settings"]
            c = S.finite_cell(sub, key)
            arrays = {f"snap_{k}": v for k, v in c.snaps.items()}
            arrays.update(npmle_weights=c.npmle_weights, npmle_curves=c.npmle_curves,
                          end_curves=c.end_curves, end_ll=c.end_ll, cap=np.array(c.cap),
                          seeds=np.array(meta["seeds"]))
            data = npz_bytes(arrays)
            rel = f"{PREFIX}/{sub}/{key}.npz"
            members[rel] = data
            src = RUNS / sub / f"{key}.npz"
            manifest["files"][rel] = {"settings": meta, "sha256": sha(data),
                                      "source": str(src.relative_to(ROOT)),
                                      "source_sha256": sha(src.read_bytes()),
                                      "shapes": {k: list(np.shape(v)) for k, v in arrays.items()}}
    for name in S.population_tasks("population"):
        t = S.population_task("population", name)
        arrays = {f"snap_{k}": v for k, v in t.snaps.items()}
        arrays["rows"] = np.array([f"{d}|{i}" for d, i in t.rows])
        data = npz_bytes(arrays)
        rel = f"{PREFIX}/population/{name}.npz"
        members[rel] = data
        src = RUNS / "population" / f"{name}.npz"
        done = json.loads((OUT / "population" / f"{name}.done.json").read_text())
        manifest["files"][rel] = {"settings": done["settings"], "sha256": sha(data),
                                  "source": str(src.relative_to(ROOT)),
                                  "source_sha256": sha(src.read_bytes()),
                                  "shapes": {k: list(np.shape(v)) for k, v in arrays.items()}}

    # bitwise check: read every bundled file back through the bundle code path
    stage = Path(os.environ.get("TMPDIR", "/tmp")) / f"{NAME}-check"
    import shutil
    shutil.rmtree(stage, ignore_errors=True)
    for rel, data in members.items():
        p = stage / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
    S.BUNDLE = stage / PREFIX
    os.environ["WBTM_EM_SNAPSHOTS"] = "bundle"
    n_checked = 0
    for rel in members:
        sub, fname = rel[len(PREFIX) + 1:].split("/")
        key = fname[:-4]
        os.environ["WBTM_EM_SNAPSHOTS"] = "runs"
        ref = S.finite_cell(sub, key) if sub.startswith("finite") else S.population_task(sub, key)
        os.environ["WBTM_EM_SNAPSHOTS"] = "bundle"
        got = S.finite_cell(sub, key) if sub.startswith("finite") else S.population_task(sub, key)
        for k in ref.snaps:
            assert np.array_equal(ref.snaps[k], got.snaps[k]), (rel, k)
        if sub.startswith("finite"):
            for f in ("npmle_weights", "npmle_curves", "end_curves", "end_ll"):
                assert np.array_equal(getattr(ref, f), getattr(got, f)), (rel, f)
        else:
            assert ref.rows == got.rows, rel
        n_checked += 1
    shutil.rmtree(stage, ignore_errors=True)
    manifest["bitwise_checked_against_runs"] = n_checked

    members[f"{PREFIX}/README.md"] = README.encode()
    members[f"{PREFIX}/MANIFEST.json"] = (json.dumps(manifest, indent=1, sort_keys=True) + "\n").encode()
    members[f"{PREFIX}/CHECKSUMS.sha256"] = "".join(
        f"{sha(d)}  {n}\n" for n, d in sorted(members.items())).encode()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    final = out / f"{NAME}.zip"
    part = final.with_suffix(".zip.part")
    with zipfile.ZipFile(part, "w", zipfile.ZIP_STORED) as z:     # npz members are already compressed
        for name, data in sorted(members.items()):
            info = zipfile.ZipInfo(name, date_time=(2026, 10, 2, 0, 0, 0))
            info.external_attr = 0o644 << 16
            z.writestr(info, data)
    part.replace(final)
    digest = sha(final.read_bytes())
    final.with_suffix(".zip.sha256").write_text(f"{digest}  {final.name}\n")
    print(f"{final}: {len(members)} files, {final.stat().st_size / 2**20:.1f} MB, sha256 {digest}; "
          f"{n_checked} files bitwise-checked against runs/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
