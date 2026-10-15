"""Experiment 8 -- minimax lower-bound hard-pair search (frontier only).

Theorem 1 (draft, Section 5).  Let q(Q) be the law of a single individual's
observable record through date t.  With

    R_i(s) = 1 - W^s(Q_i) / W^{t*(Q_i)}(Q_i)          relative welfare regret
    zeta   = min_s [ R_0(s) + R_1(s) ]

if KL(q(Q_0) || q(Q_1)) <= c/N then every timing rule measurable with respect to
N such records incurs expected relative regret at least (1/8) e^{-c} zeta under
at least one of the two.

zeta is the least *total* regret from committing to any single date across both
instances.  It is deliberately not the gap between the two optima: a rule may
return any date, and if some third date is near-optimal under both then zeta
vanishes and no bound is available -- correctly.  The fraction of different-t*
pairs for which that happens is reported.

**The conversion to an implied sample size is not performed here.**  Inverting
Theorem 1 requires a target regret threshold eps,

    N >= log( zeta / (8 eps) ) / KL( q(Q_0) || q(Q_1) ),

and eps is not stated in the draft (NEEDS_DECISION D2b).  This module saves the
(KL, zeta) frontier only; the Table 5 sample-size row stays BLOCKED.

Every candidate pair is saved, not only the frontier.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from wtm.distributions import BetaRisk  # noqa: E402
from wtm.lp import support_grid  # noqa: E402
from wtm.observation import individual_functionals, phi, utility_poly  # noqa: E402
from wtm.welfare import fill_greedy  # noqa: E402

ZERO_ZETA = 1e-4          # "zeta ~ 0": some third date is near-optimal under both


class Evaluator:
    """Trajectories and record laws for grid-supported distributions."""

    def __init__(self, grid, gamma: int, T: int, beta: float, t: int):
        self.grid, self.gamma, self.T, self.beta, self.t = grid, gamma, T, beta, t
        self.dates = np.arange(1, T)
        self.PHI, self.UTIL = {}, {}
        for s in self.dates:
            s = int(s)
            u = utility_poly(s, T).evaluate(grid)
            P = np.vstack([phi(s, k, gamma).evaluate(grid) for k in range(s + 1)])
            self.PHI[s], self.UTIL[s] = P, P * u[None, :]
        self.atoms = individual_functionals(t, gamma)
        self.REC = np.vstack([f.evaluate(grid) for _, f in self.atoms])

    def trajectory(self, w):
        return np.array([fill_greedy(self.PHI[int(s)] @ w, self.UTIL[int(s)] @ w,
                                     self.beta)[0] for s in self.dates])

    def regret(self, W):
        m = W.max()
        return (1.0 - W / m) if m > 0 else np.full_like(W, np.nan)

    def record_law(self, w):
        q = self.REC @ w
        return q / q.sum()


def kl(p, q):
    """KL(p||q) over a finite alphabet, with support mismatch reported as inf."""
    p = np.asarray(p, float)
    q = np.asarray(q, float)
    mask = p > 0
    if np.any(q[mask] <= 0):
        return np.inf
    return float(np.sum(p[mask] * np.log(p[mask] / q[mask])))


def candidate_pool(grid, rng, n_beta=90, n_mixture=70, n_discrete=70):
    """Broad candidate classes, all represented as weights on the common grid."""
    out = []

    def add(w, kind, params):
        w = np.maximum(np.asarray(w, float), 0.0)
        tot = w.sum()
        if tot <= 0 or not np.isfinite(tot):
            return
        out.append((w / tot, kind, params))

    def beta_weights(a, b):
        dens = np.exp((a - 1) * np.log(np.clip(grid, 1e-12, 1))
                      + (b - 1) * np.log(np.clip(1 - grid, 1e-12, 1)))
        return dens

    for a, b in ((1, 32), (1, 8), (1, 4)):                 # benchmark families
        add(beta_weights(a, b), "beta", f"a={a};b={b}")
    for _ in range(n_beta):
        a = float(np.exp(rng.uniform(np.log(0.3), np.log(5.0))))
        b = float(np.exp(rng.uniform(np.log(1.0), np.log(60.0))))
        add(beta_weights(a, b), "beta", f"a={a:.4f};b={b:.4f}")
    for _ in range(n_mixture):
        a1 = float(np.exp(rng.uniform(np.log(0.3), np.log(5.0))))
        b1 = float(np.exp(rng.uniform(np.log(1.0), np.log(60.0))))
        a2 = float(np.exp(rng.uniform(np.log(0.3), np.log(5.0))))
        b2 = float(np.exp(rng.uniform(np.log(1.0), np.log(60.0))))
        pi = float(rng.uniform(0.02, 0.5))
        w1, w2 = beta_weights(a1, b1), beta_weights(a2, b2)
        add((1 - pi) * w1 / w1.sum() + pi * w2 / w2.sum(), "mixture",
            f"a1={a1:.3f};b1={b1:.3f};a2={a2:.3f};b2={b2:.3f};pi={pi:.3f}")
    for _ in range(n_discrete):
        k = int(rng.integers(2, 6))
        idx = rng.choice(len(grid), size=k, replace=False)
        w = np.zeros(len(grid))
        w[idx] = rng.dirichlet(np.ones(k))
        add(w, "discrete", ";".join(f"{grid[i]:.4f}:{w[i]:.4f}" for i in idx))
    return out


def _pair_row(ev, i, j, pool, info):
    wi, wj = pool[i][0], pool[j][0]
    Ri, Rj = info[i]["R"], info[j]["R"]
    total = Ri + Rj
    s_idx = int(np.argmin(total))
    zeta = float(total[s_idx])
    qi, qj = info[i]["q"], info[j]["q"]
    fwd, rev = kl(qi, qj), kl(qj, qi)
    sym = 0.5 * (fwd + rev) if np.isfinite(fwd) and np.isfinite(rev) else np.inf
    return {
        "t_star_0": info[i]["t_star"], "t_star_1": info[j]["t_star"],
        "class_0": pool[i][1], "class_1": pool[j][1],
        "params_0": pool[i][2], "params_1": pool[j][2],
        "zeta": zeta,
        "argmin_date": int(ev.dates[s_idx]),
        "zeta_is_negligible": bool(zeta < ZERO_ZETA),
        "kl_forward": fwd, "kl_reverse": rev, "kl_symmetrized": sym,
        "regret_0_at_argmin": float(Ri[s_idx]),
        "regret_1_at_argmin": float(Rj[s_idx]),
        "t_star_separation": abs(info[i]["t_star"] - info[j]["t_star"]),
        "W_max_0": info[i]["Wmax"], "W_max_1": info[j]["Wmax"],
    }


def pareto_frontier(df: pd.DataFrame) -> pd.DataFrame:
    """Pairs not dominated on (low symmetrized KL, high zeta)."""
    sub = df[np.isfinite(df.kl_symmetrized) & (df.zeta > 0)].copy()
    if sub.empty:
        return sub
    sub = sub.sort_values("kl_symmetrized")
    keep, best = [], -np.inf
    for r in sub.itertuples():
        if r.zeta > best:
            keep.append(r.Index)
            best = r.zeta
    return sub.loc[keep]


def run_config(gamma, T, beta, t, grid, seed, max_pairs=200_000):
    rng = np.random.default_rng(seed)
    ev = Evaluator(grid, gamma, T, beta, t)
    pool = candidate_pool(grid, rng)

    info = []
    for w, _, _ in pool:
        W = ev.trajectory(w)
        info.append({"R": ev.regret(W), "t_star": int(ev.dates[int(np.argmax(W))]),
                     "q": ev.record_law(w), "Wmax": float(W.max())})

    rows, n_diff = [], 0
    n = len(pool)
    for i in range(n):
        for j in range(i + 1, n):
            if info[i]["t_star"] == info[j]["t_star"]:
                continue
            n_diff += 1
            if len(rows) >= max_pairs:
                continue
            rows.append(_pair_row(ev, i, j, pool, info))

    df = pd.DataFrame(rows)
    if not df.empty:
        df.insert(0, "prefix_t", t)
        df.insert(0, "beta", beta)
        df.insert(0, "T", T)
        df.insert(0, "gamma", gamma)
        df.insert(0, "search_seed", seed)
        df["n_candidates"] = n
        df["n_pairs_distinct_tstar"] = n_diff
    return df, len(pool), n_diff


def run(gammas=(1,), horizons=(12, 20), betas=(0.0005, 0.002, 0.01),
        prefixes=(2, 3, 4), grid_size=201, seed=20260821):
    grid = support_grid(grid_size)
    all_pairs, summary = [], []
    for gamma in gammas:
        for T in horizons:
            for beta in betas:
                for t in prefixes:
                    t0 = time.perf_counter()
                    df, n_cand, n_diff = run_config(gamma, T, beta, t, grid,
                                                    seed + 977 * t + 31 * T)
                    all_pairs.append(df)
                    fr = pareto_frontier(df)
                    summary.append({
                        "gamma": gamma, "T": T, "beta": beta, "prefix_t": t,
                        "grid_resolution": grid_size,
                        "n_candidates": n_cand,
                        "n_pairs_distinct_tstar": n_diff,
                        "n_pairs_recorded": len(df),
                        "n_frontier": len(fr),
                        "max_zeta": float(df.zeta.max()) if len(df) else np.nan,
                        "median_zeta": float(df.zeta.median()) if len(df) else np.nan,
                        "frac_zeta_negligible": (float(df.zeta_is_negligible.mean())
                                                 if len(df) else np.nan),
                        "min_symmetrized_kl": (float(df.kl_symmetrized.min())
                                               if len(df) else np.nan),
                        "n_infinite_kl": (int((~np.isfinite(df.kl_symmetrized)).sum())
                                          if len(df) else 0),
                        "runtime_s": time.perf_counter() - t0,
                    })
    pairs = pd.concat(all_pairs, ignore_index=True) if all_pairs else pd.DataFrame()
    frontier = (pd.concat([pareto_frontier(d) for d in all_pairs if not d.empty],
                          ignore_index=True) if all_pairs else pd.DataFrame())
    return pairs, frontier, pd.DataFrame(summary)


def main():
    results = ROOT / "results"
    results.mkdir(parents=True, exist_ok=True)
    pairs, frontier, summary = run()
    pairs.to_csv(results / "lower_bound_pairs.csv", index=False)
    frontier.to_csv(results / "lower_bound_frontier.csv", index=False)
    summary.to_csv(results / "lower_bound_summary.csv", index=False)
    pd.set_option("display.width", 220)
    print(summary.to_string(index=False))
    print("\nSample-size conversion NOT performed: needs target regret eps "
          "(NEEDS_DECISION D2b).  N >= log(zeta/(8 eps)) / KL.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
