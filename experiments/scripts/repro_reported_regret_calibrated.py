"""Targeted reproduction of the reported bagging and regret-calibration rows.

The manuscript's "Bagged NPMLE" and "Regret-calibrated" rows are produced by
`experiments/scripts/bagged_full_b200.py`, which evaluates both rules on one
shared bootstrap ensemble per run:

    bagged : argmax_t  sum_b W_b^t
    RC     : argmax_t  sum_b W_b^t / M_b,   M_b = max_u W_b^u
             (= `wtm.regret_calibrated.expected_regret_rule`, argmin of mean
              relative regret; ties to the earliest date)

The separately implemented `wtm.regret_calibrated.regret_calibrated()` is a
different, randomized sequential procedure and does NOT generate the reported
row.  This script checks that claim mechanically.

Stage 1 (always, ~1 min): for every archived run in
`results/bagged_full_b200_raw.csv`, reload the stored bootstrap trajectories
from `results/bagged_full_b200_trajectories/`, recompute both decisions and the
retained-draw count, and require exact agreement with the archived rows.

Stage 2 (`--refit K`, default 0): re-simulate K randomly chosen runs from their
frozen seeds, refit the primary NPMLE and all 200 bootstrap fits from scratch,
and require that the single-fit, bagged and RC dates and the retained count
match the archive.  Stage 2 at K=20 takes a few minutes on one core.

Finally prints the Table 3 bagging / RC cells on the matched tau=2 sample.

Usage:
    python experiments/scripts/repro_reported_regret_calibrated.py [--refit K]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
RESULTS = ROOT / "results"
TRAJ = RESULTS / "bagged_full_b200_trajectories"
KEY = ["cell_id", "N", "prefix_t", "seed_index"]

from wtm.regret_calibrated import expected_regret_rule  # noqa: E402
from experiments.scripts import bagged_full_b200 as b200  # noqa: E402
from experiments.scripts.bagged_full_b200 import bagged_date, run_cell, single_date  # noqa: E402


def stage1(raw: pd.DataFrame) -> None:
    runs = raw[raw.SKIP.isna()].pivot_table(
        index=KEY, columns="rule", values="selected_date").reset_index()
    used = raw[raw.rule == "bagged"].set_index(KEY).n_bootstrap_used
    bad = missing = 0
    for r in runs.itertuples(index=False):
        path = TRAJ / f"{r.cell_id}_{int(r.seed_index)}.npz"
        if not path.exists():
            # the packaged supplement ships only the Table 3 runs
            # (tau = 2, N in {1e3, 1e5}); the repository has all of them
            missing += 1
            continue
        z = np.load(path)
        Wb = z["W_b"]
        # the equivalent weighted-sum form stated in the paper
        M = Wb.max(axis=1, keepdims=True)
        d_rc_sum = int(np.argmax((Wb / M).sum(axis=0))) + 1
        d_rc, _ = expected_regret_rule(Wb)
        ok = (single_date(z["W_single"]) == r.single
              and bagged_date(Wb) == r.bagged
              and d_rc == r.rc
              and Wb.shape[0] == used.loc[(r.cell_id, r.N, r.prefix_t, r.seed_index)])
        # argmin mean regret and argmax sum W/M coincide up to float ties
        ok &= d_rc_sum == d_rc or np.isclose(
            (Wb / M).sum(axis=0)[d_rc - 1], (Wb / M).sum(axis=0).max(), rtol=1e-6)
        bad += not ok
    checked = len(runs) - missing
    print(f"stage 1: {len(runs)} archived runs, {checked - bad} of {checked} with cached "
          f"trajectories reproduced exactly, {bad} mismatches"
          + (f" ({missing} runs without packaged trajectories skipped)" if missing else ""))
    if bad or not checked:
        raise SystemExit(1)


def stage2(raw: pd.DataFrame, k: int, seed: int) -> None:
    import tempfile
    from wtm.config import load_frozen, population_cells
    # run_cell writes trajectories; keep the archive read-only
    b200.TRAJ_DIR = Path(tempfile.mkdtemp(prefix="wtm_refit_"))
    ok_rows = raw[raw.SKIP.isna() & (raw.rule == "bagged")]
    pick = ok_rows.sample(n=k, random_state=seed)
    cells = {(c.dist_name, c.T, c.beta): c for c in population_cells(load_frozen()) if c.gamma == 1}
    bad = 0
    for r in pick.itertuples(index=False):
        c = cells[(r.distribution, r.T, r.beta)]
        rows = run_cell((c.dist_name, 1, c.T, c.beta, int(r.N), int(r.prefix_t)))
        new = {(x["rule"]): x for x in rows
               if x.get("seed_index") == r.seed_index and "rule" in x}
        arch = raw[(raw.cell_id == r.cell_id) & (raw.seed_index == r.seed_index)].set_index("rule")
        same = all(int(new[k_]["selected_date"]) == int(arch.loc[k_, "selected_date"])
                   for k_ in ("single", "bagged", "rc"))
        same &= int(new["bagged"]["n_bootstrap_used"]) == int(arch.loc["bagged", "n_bootstrap_used"])
        bad += not same
        print(f"  refit {r.cell_id} seed {r.seed_index}: {'match' if same else 'MISMATCH'}", flush=True)
    print(f"stage 2: {k - bad}/{k} runs refit from scratch match the archive")
    if bad:
        raise SystemExit(1)


def table3_cells(raw: pd.DataFrame) -> None:
    d = raw[(raw.prefix_t == 2) & raw.N.isin([1000, 100000]) & raw.SKIP.isna()]
    print("\nTable 3 cells (tau=2; MSR x 1e3, p10):")
    for N in (1000, 100000):
        for rule, label in (("single", "Certified NPMLE"), ("bagged", "Bagged NPMLE"),
                            ("rc", "Regret-calibrated")):
            g = d[(d.N == N) & (d.rule == rule)]
            print(f"  N={N:>6} {label:<18} n={len(g)} MSR={1e3 * g.squared_regret.mean():.2f} "
                  f"p10={np.percentile(g.welfare_efficiency, 10):.4f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=20260906)
    a = ap.parse_args()
    raw = pd.read_csv(RESULTS / "bagged_full_b200_raw.csv")
    stage1(raw)
    if a.refit:
        stage2(raw, a.refit, a.seed)
    table3_cells(raw)


if __name__ == "__main__":
    main()
