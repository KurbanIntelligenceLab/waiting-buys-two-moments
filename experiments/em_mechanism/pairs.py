"""Exact compatible pairs (the Table 1 / Appendix D.3 extreme pairs), cached.

For tau = 2 this is `experiments/scripts/matched_info_pairs.py::exact_pairs`
(reconstruction-v1's pair definitions) verbatim in substance; for tau = 3, 4
the same construction (`extreme_pair_exact.py`) at the longer prefix.  Members
are exact rational members of M_tau on the 2,001-point grid; their regret
vectors are computed in exact arithmetic and stored as floats.

Both members match the reference population's moments through degree
2*tau - 1, so they induce exactly the reference population's pre-failure
observation law.  `check_shared_law` verifies this numerically.

Usage:  python -m experiments.em_mechanism.pairs [--tau 2 3 4]
Writes  results/em_mechanism/frozen/pairs_tau{tau}.json (once; reused after).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from experiments.em_mechanism.common import ROOT  # noqa: E402  (honours WBTM_ROOT)

FROZEN = ROOT / "results" / "em_mechanism" / "frozen"


def build_pairs(tau: int) -> list:
    from wtm.config import config_id, load_frozen, population_cells
    from wtm.decision_id import DecisionProblem
    from wtm.lp import IdentifiedSet, support_grid
    argv, sys.argv = sys.argv, sys.argv[:1]     # extreme_pair_rebuild parses argv at import
    try:
        from experiments.scripts.extreme_pair_exact import exact_member, exact_trajectory
        from experiments.scripts.extreme_pair_rebuild import GRID_SIZE, extreme_members
    finally:
        sys.argv = argv
    cells = [c for c in population_cells(load_frozen()) if c.gamma == 1]
    assert len(cells) == 30
    grid = support_grid(GRID_SIZE)
    out = []
    for cell in cells:
        prob = DecisionProblem(cell.dist, cell.T, cell.beta, tau)
        S = IdentifiedSet(cell.dist, tau, grid)
        lo, hi = extreme_members(grid, S, cell.T)
        assert lo.success and hi.success
        members = [exact_member(prob, lo.x), exact_member(prob, hi.x)]
        assert all(wi >= 0 for _, w in members for wi in w)
        regrets, welfare = [], []
        for atoms, w in members:
            W = exact_trajectory(prob, atoms, w)
            best = max(W.values())
            regrets.append([float(1 - W[s] / best) for s in prob.dates])
            welfare.append([float(W[s]) for s in prob.dates])
        zeta = min(regrets[0][i] + regrets[1][i] for i in range(len(prob.dates)))
        out.append({
            "config_id": config_id(dist=cell.dist_name, gamma=1, T=cell.T, beta=cell.beta),
            "distribution": cell.dist_name, "T": cell.T, "beta": cell.beta, "tau": tau,
            "dates": [int(s) for s in prob.dates],
            "members": [{"atoms": [float(a) for a in atoms],
                         "atoms_exact": [str(a) for a in atoms],
                         "weights": [float(x) for x in w],
                         "weights_exact": [str(x) for x in w]} for atoms, w in members],
            "regret": regrets, "welfare": welfare, "zeta_over_2": zeta / 2,
        })
    return out


def load_pairs(tau: int) -> list:
    path = FROZEN / f"pairs_tau{tau}.json"
    if not path.exists():
        FROZEN.mkdir(parents=True, exist_ok=True)
        t0 = time.perf_counter()
        pairs = build_pairs(tau)
        path.write_text(json.dumps(pairs, indent=1) + "\n")
        print(f"built {len(pairs)} tau={tau} pairs in {time.perf_counter() - t0:.0f}s -> {path}")
    return json.loads(path.read_text())


def member_dist(pair: dict, i: int):
    from wtm.distributions import DiscreteRisk
    m = pair["members"][i]
    w = np.asarray(m["weights"], dtype=float)
    return DiscreteRisk(points=tuple(m["atoms"]), weights=tuple(w / w.sum()),
                        name=f"pair_{'lower' if i == 0 else 'upper'}_{pair['config_id']}")


def check_shared_law(pair: dict, ref_dist) -> dict:
    """Max |q_member - q_ref| over pre-failure record cells, for both members."""
    from experiments.em_mechanism.core import exact_cell_probs
    q = exact_cell_probs(ref_dist, pair["tau"])
    d = [float(np.abs(exact_cell_probs(member_dist(pair, i), pair["tau"]) - q).max())
         for i in (0, 1)]
    return {"max_abs_diff_lower": d[0], "max_abs_diff_upper": d[1]}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tau", type=int, nargs="+", default=[2, 3, 4])
    a = ap.parse_args()
    for t in a.tau:
        load_pairs(t)
