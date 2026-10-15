"""Reproducible examples for results/em_mechanism/AUDIT.md (each prints its evidence).

  A2  a scored "certified" NPMLE fit whose KKT residual exceeds 1e-8
  A3  converged, certified EM from two initializations: different weights and
      selected date at essentially equal objective (fit_npmle docstring claim)
  A4  the support-reduction solver does not reach tighter tolerances, and the
      selected date moves with the tolerance
  A5  explicit witness: a weight vector with converged EM's fitted cell
      probabilities (same objective and KKT residual) selecting a different date

Usage: python -m experiments.em_mechanism.audit_examples
"""
from __future__ import annotations

import numpy as np

from experiments.em_mechanism.common import gamma1_cells
from experiments.em_mechanism.core import (Welfare, certified_npmle, exact_cell_probs, grid_and_psi,
                                           init_weights, run_em)


def a2():
    from experiments.em_mechanism.pairs import load_pairs
    from experiments.em_mechanism.run_finite import simulate_counts
    cell = next(c for c in gamma1_cells() if c.cid == "82d22f34a4dc")
    pair = next(p for p in load_pairs(2) if p["config_id"] == cell.cid)
    counts, sims = simulate_counts("ref", cell, pair, 100000, [1015])
    grid, Psi = grid_and_psi(200, 2)
    f = certified_npmle(counts[0], grid, 2, Psi)
    print(f"A2 {cell.dist_name} T={cell.T} beta={cell.beta} N=1e5 seed 1015 (sim_seed {sims[0]}): "
          f"kkt={f.kkt:.3e} (> 1e-8), support={f.n_support}, status={f.status}")


def a3_a5():
    from experiments.em_mechanism.maximizers import ll_kkt, witness
    cell = next(c for c in gamma1_cells() if c.dist_name == "beta_1_32" and c.T == 20 and c.beta == 0.01)
    grid, Psi = grid_and_psi(200, 2)
    q = exact_cell_probs(cell.dist, 2)
    ev = Welfare(grid, cell.T, cell.beta)
    W0 = np.vstack([init_weights("uniform", grid), init_weights("rand1", grid)])
    cap = 10_000_000
    r = run_em(np.vstack([q, q]), Psi, grid, W0, np.array([0, cap]), {"e": ev}, snapshot_at=[cap])
    w_u, w_r = r.snapshots[cap]
    for tag, w in (("uniform", w_u), ("rand1", w_r)):
        ll, kkt = ll_kkt(w, q, Psi)
        d = ev.decide(w[None], ev.curves(w[None]))[0]
        print(f"A3 EM[{tag}] after {cap:.0e} updates: ll={ll:.15f} kkt={kkt:.2e} date={d}")
    print(f"A3 max|qhat_uniform - qhat_rand1| = {np.abs(Psi @ w_u - Psi @ w_r).max():.2e}, "
          f"weight TV = {0.5 * np.abs(w_u - w_r).sum():.3f}")
    for tol in (1e-8, 1e-11, 1e-13):
        f = certified_npmle(q, grid, 2, Psi, delta=tol)
        d = ev.decide(f.weights[None], ev.curves(f.weights[None]))[0]
        print(f"A4 NPMLE delta={tol:.0e}: ll={f.ll:.15f} kkt={f.kkt:.2e} status={f.status} date={d}")
    f = certified_npmle(q, grid, 2, Psi, delta=1e-13)
    w_wit, _ = witness(f.weights, Psi @ w_u, Psi)
    ll, kkt = ll_kkt(w_wit, q, Psi)
    d = ev.decide(w_wit[None], ev.curves(w_wit[None]))[0]
    print(f"A5 witness with EM[uniform]'s cell probabilities: max resid "
          f"{np.abs(Psi @ w_wit - Psi @ w_u).max():.1e}, ll={ll:.15f}, kkt={kkt:.2e}, date={d}")


if __name__ == "__main__":
    a2()
    a3_a5()
