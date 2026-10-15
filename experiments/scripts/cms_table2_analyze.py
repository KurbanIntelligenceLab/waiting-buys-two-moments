"""Build Table 2 with CMS added, on exactly the sample the table uses.

THE SAMPLE. Table 2 (and Table 8) are tau = 2, reported at N = 10^3 and
N = 10^5, pooled over all 30 configurations and all 20 seeds: 600 runs per
column. Fifteen of the thirty configurations are at T = 20.

Every procedure is scored on the SAME runs. Where a procedure has no decision
for a run, that run is dropped from every procedure's column, so the comparison
stays matched rather than each row using whatever it happens to have.

MSR gain is the reduction relative to the certified single-fit NPMLE, in the
paper's convention: positive favours the row.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
RESULTS = ROOT / "results"
CKPT = RESULTS / "cms_table2_cells"
DRAWS = RESULTS / "cms_table2_draws.jsonl.gz"
SUFFIX = ""          # "_prefailure" with --resampling prefailure
KEY = ["cell_id", "N", "prefix_t", "seed_index"]
NS = [1000, 100000]
TAU = 2

ROWS = [
    ("Oracle", "main_benchmark_raw_all.csv", "oracle"),
    ("Act immediately", "main_benchmark_raw_all.csv", "act_immediately"),
    ("Certified NPMLE", "main_benchmark_raw_all.csv", "npmle"),
    ("Bagged NPMLE", "bagged_full_b200_raw.csv", "bagged"),
    ("Regret-calibrated", "main_benchmark_raw_all.csv", "regret_calibrated"),
    ("Early-stopped EM (55 it.)", "early_stopped_npmle_raw.csv", "early_stopped_npmle"),
]


def draw_records():
    """Per-draw CMS records: the checkpoint files, or their consolidated copy.

    The publication checkout keeps the 24,000 records in one gzipped
    JSON-lines file. A fresh fitting run may additionally produce per-draw
    checkpoints; those take precedence when present.
    """
    files = sorted(CKPT.glob("*.json"))
    if files:
        for f in files:
            yield from json.loads(f.read_text())
    else:
        import gzip
        with gzip.open(DRAWS, "rt") as fh:
            for line in fh:
                yield json.loads(line)


def cms_decisions():
    """One row per run: the CMS date from averaging max risk over its draws."""
    from wtm.config import load_frozen, population_cells
    from wtm.rules import evaluate
    from wtm.welfare import PopulationConfig

    by_run = defaultdict(list)
    for r in draw_records():
        by_run[(r["distribution"], r["T"], r["beta"], r["N"],
                r["seed_index"])].append(r)

    cfg = load_frozen()
    cells = {(c.dist_name, c.T, c.beta): c
             for c in population_cells(cfg) if c.gamma == 1}
    out, nested = [], []
    for key, rs in by_run.items():
        ok = sorted([r for r in rs if r["success"]], key=lambda r: r["draw"])
        if not ok:
            continue
        risks = [json.loads(r["max_risk"]) for r in ok]
        dates = sorted(risks[0], key=int)
        avg = {d: float(np.mean([x[d] for x in risks])) for d in dates}
        sel = int(min(avg, key=lambda d: (avg[d], int(d))))
        cell = cells[(key[0], key[1], key[2])]
        pop = PopulationConfig(cell.dist, 1, key[1], key[2])
        out.append({"cell_id": ok[0]["cell_id"], "N": key[3], "prefix_t": TAU,
                    "seed_index": key[4], "T": key[1], "distribution": key[0],
                    "n_draws": len(ok), **evaluate(sel, pop)})
        for B in (5, 10, 20):
            if B <= len(risks):
                a = {d: float(np.mean([x[d] for x in risks[:B]])) for d in dates}
                nested.append({"cell_id": ok[0]["cell_id"], "N": key[3],
                               "seed_index": key[4], "B": B,
                               "selected_date": int(min(a, key=lambda d: (a[d], int(d))))})
    return pd.DataFrame(out), pd.DataFrame(nested)


def main() -> None:
    import argparse
    global CKPT, DRAWS, SUFFIX
    ap = argparse.ArgumentParser()
    ap.add_argument("--resampling", choices=["paths", "prefailure"], default="paths",
                    help="paths (default): the archived Table 3 run")
    ap.add_argument("--rng", choices=["legacy", "independent"], default="legacy",
                    help="legacy (default): the archived seeds")
    a = ap.parse_args()
    if (a.resampling, a.rng) != ("paths", "legacy"):
        from experiments.scripts.cms_table2_run import ckpt_dir
        CKPT = ckpt_dir(a.resampling, a.rng)
        SUFFIX = ("_prefailure" if a.rng == "legacy"
                  else f"_{a.resampling}_{a.rng}")
        DRAWS = RESULTS / f"cms_table2{SUFFIX}_draws.jsonl.gz"
    cms, nested = cms_decisions()
    cms.to_csv(RESULTS / f"cms_table2{SUFFIX}_raw.csv", index=False)
    nested.to_csv(RESULTS / f"cms_table2{SUFFIX}_nestedB.csv", index=False)
    full = cms[cms.n_draws >= 20]
    print(f"CMS runs scored: {len(cms)} ({len(full)} with all 20 draws) of 1200")
    if len(cms) < 1200:
        print("  INCOMPLETE -- run again when the box finishes")

    frames = {"CMS bootstrap (B=20)": cms}
    for label, fname, rule in ROWS:
        f = RESULTS / fname
        if not f.exists():
            continue
        d = pd.read_csv(f)
        d = d[(d.rule == rule) & (d.prefix_t == TAU) & (d.N.isin(NS))]
        if "success" in d:
            d = d[d.success.astype(str) == "True"]
        frames[label] = d

    # matched sample: runs where EVERY procedure has a decision
    common = None
    for d in frames.values():
        s = set(map(tuple, d[KEY].values))
        common = s if common is None else common & s
    sub = pd.DataFrame(list(common), columns=KEY)
    print(f"matched sample: {len(sub)} runs "
          f"({(sub.N == 1000).sum()} at 1e3, {(sub.N == 100000).sum()} at 1e5)")

    base = {}
    for N in NS:
        g = frames["Certified NPMLE"].merge(sub[sub.N == N], on=KEY)
        base[N] = g.squared_regret.mean()

    print(f"\n{'Procedure':<28}" + "".join(
        f"{c:>12}" for c in ("MSR 1e3", "gain", "p10 1e3", "MSR 1e5", "gain", "p10 1e5")))
    order = ["Oracle", "Act immediately", "Certified NPMLE", "Bagged NPMLE",
             "Regret-calibrated", "CMS bootstrap (B=20)", "Early-stopped EM (55 it.)"]
    for label in order:
        if label not in frames:
            continue
        cells = []
        for N in NS:
            g = frames[label].merge(sub[sub.N == N], on=KEY)
            if not len(g):
                cells += ["--"] * 3
                continue
            msr = g.squared_regret.mean()
            cells += [f"{1e3*msr:.2f}", f"{1e3*(base[N]-msr):+.2f}",
                      f"{np.percentile(g.welfare_efficiency, 10):.4f}"]
        print(f"{label:<28}" + "".join(f"{c:>12}" for c in cells))
    print("\nMSR in units of 1e-3; gain is the reduction against the certified NPMLE.")

    if len(nested):
        piv = nested.pivot_table(index=["cell_id", "N", "seed_index"],
                                 columns="B", values="selected_date")
        print("\nB sensitivity (share of runs whose date matches B=20):")
        for B in (5, 10):
            if B in piv.columns and 20 in piv.columns:
                print(f"  B={B:2d}: {100*(piv[B] == piv[20]).mean():.1f}%")


if __name__ == "__main__":
    main()
