"""Reachable optimal dates inside an identified set (Experiments 5 and 6).

Question: can two distributions that agree on every moment a length-t prefix
identifies still disagree about *when* to allocate?

For each candidate date s we ascend

    f_s(Q) = W^s(Q) - max_{s' != s} W^{s'}(Q)

over Q in M_t.  If f_s(Q) > 0 anywhere then s is the argmax for that Q and s is
reachable; Q is retained as a witness.  Ascent runs in the null space of the
moment constraints, so every iterate satisfies the moment conditions exactly (up
to the residual we record) and only nonnegativity has to be enforced by the step
limit.

Subgradients come from the dual form of the inner allocation LP,

    W^s(Q) = min_{lambda>=0} [ beta*lambda + sum_k (v_k(Q) - lambda m_k(Q))_+ ],

so by Danskin, at the optimal lambda*,

    dW^s/dw = sum_{k : v_k > lambda* m_k} ( UTIL[k,:] - lambda* PHI[k,:] ).

W^s is convex in Q, so f_s is a difference of convex functions: ascent finds
local optima and the search is a *constructive lower bound* on the reachable set
|T_t|, never a proof of uniqueness.  Experiment 4's LP bounds supply the
complementary certified *upper* bound -- if max_Q W^s < min_Q W^{s'} for some
s', date s is certifiably unreachable -- and the two are reported together.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linprog

from .lp import SOLVER, group_matrices
from .welfare import fill_greedy


@dataclass
class SearchStats:
    attempts: int = 0
    feasible_candidates: int = 0
    iterations: int = 0
    best_gap: float = -np.inf
    moment_residual: float = np.nan
    min_mass: float = np.nan
    status: str = "ok"


class ReachableDateSearch:
    """Search M_t for distributions whose optimal allocation date is s."""

    def __init__(self, identified_set, gamma: int, T: int, beta: float,
                 dates=None):
        self.S = identified_set
        self.grid = identified_set.grid
        self.gamma, self.T, self.beta = int(gamma), int(T), float(beta)
        self.dates = np.arange(1, T) if dates is None else np.asarray(dates)
        self._cache = {int(s): group_matrices(self.grid, int(s), gamma, T)
                       for s in self.dates}
        # Projector onto the null space of the moment constraints.  Formed as
        # I - A^T (A A^T)^+ A and applied factored, so we never materialise the
        # G x (G-2t) null-space basis (G is up to 2001) and never invert the
        # ill-conditioned Vandermonde A directly.
        A = self.S.A
        self._proj = A.T @ np.linalg.pinv(A @ A.T)   # G x 2t

    # -- welfare and subgradient --------------------------------------------
    def welfare(self, w, s: int) -> float:
        PHI, UTIL = self._cache[int(s)]
        return fill_greedy(PHI @ w, UTIL @ w, self.beta)[0]

    def trajectory(self, w) -> np.ndarray:
        return np.array([self.welfare(w, int(s)) for s in self.dates])

    def subgradient(self, w, s: int) -> np.ndarray:
        """dW^s/dw by the envelope theorem for the inner allocation LP.

        With L = sum_k a_k v_k(w) - lambda (sum_k a_k m_k(w) - beta) at the
        optimal (a, lambda),

            dW^s/dw = sum_k a_k ( UTIL[k,:] - lambda PHI[k,:] ),

        weighting each group by its *fill fraction*.  The dual form would weight
        by the indicator 1{v_k > lambda m_k}, which is also a valid subgradient
        but silently drops the marginal group -- and the marginal group is the
        only one contributing whenever the budget runs out inside the top
        y-group, which is the common case at these budget ratios.  That version
        returns a zero direction and the ascent stalls at its starting point.
        """
        PHI, UTIL = self._cache[int(s)]
        m, v = PHI @ w, UTIL @ w
        _, a = fill_greedy(m, v, self.beta)
        frac = np.where((a > 1e-12) & (a < 1 - 1e-12))[0]
        if len(frac):
            k = int(frac[0])
            lam = v[k] / m[k] if m[k] > 0 else 0.0
        else:
            on = np.where(a > 1e-12)[0]
            lam = (v[on.min()] / m[on.min()]) if len(on) and m[on.min()] > 0 else 0.0
        return (a[:, None] * (UTIL - lam * PHI)).sum(axis=0)

    # -- feasible starting points -------------------------------------------
    def random_start(self, rng: np.random.Generator):
        """A random vertex of M_t (LP with a random objective)."""
        c = rng.standard_normal(self.S.n)
        res = linprog(c=c, A_eq=self.S.A, b_eq=self.S.b,
                      bounds=[(0.0, None)] * self.S.n, method=SOLVER)
        return (res.x if res.success else None)

    # -- ascent --------------------------------------------------------------
    def ascend(self, w0, s: int, max_iter: int = 400, step_init: float = 0.25):
        """Ascend f_s = W^s - max_{s'!=s} W^{s'} from w0.  Returns (w, gap, iters)."""
        w = np.array(w0, dtype=float)
        others = [int(x) for x in self.dates if int(x) != int(s)]

        def objective(x):
            wo = max(self.welfare(x, o) for o in others)
            return self.welfare(x, int(s)) - wo, wo

        best_val, _ = objective(w)
        best_w = w.copy()
        step = float(step_init)
        iters = 0
        for _ in range(int(max_iter)):
            iters += 1
            _, wo = objective(w)
            rival = max(others, key=lambda o: self.welfare(w, o))
            d = self.subgradient(w, int(s)) - self.subgradient(w, rival)
            d = d - self._proj @ (self.S.A @ d)    # stay on the moment manifold
            nd = np.linalg.norm(d)
            if nd < 1e-18:
                break
            d = d / nd
            # largest step that keeps every mass nonnegative
            neg = d < -1e-15
            cap = np.min(-w[neg] / d[neg]) if np.any(neg) else np.inf
            trial = min(step, 0.99 * cap) if np.isfinite(cap) else step
            improved = False
            while trial > 1e-14:
                cand = w + trial * d
                cand = np.maximum(cand, 0.0)
                val, _ = objective(cand)
                if val > best_val + 1e-18:
                    w, best_val, best_w = cand, val, cand.copy()
                    improved = True
                    break
                trial *= 0.5
            if not improved:
                step *= 0.5
                if step < 1e-12:
                    break
        return best_w, best_val, iters

    def reachable(self, s: int, n_restarts: int = 24, max_iter: int = 400,
                  step_init: float = 0.25, seeds=None, rng=None):
        """Try to certify date s as reachable.  Returns (witness, stats)."""
        rng = rng or np.random.default_rng(0)
        stats = SearchStats()
        witness = None
        starts = list(seeds or [])
        for _ in range(int(n_restarts)):
            w0 = self.random_start(rng)
            if w0 is not None:
                starts.append(w0)
        for w0 in starts:
            stats.attempts += 1
            if w0 is None:
                continue
            stats.feasible_candidates += 1
            w, gap, iters = self.ascend(w0, s, max_iter, step_init)
            stats.iterations += iters
            if gap > stats.best_gap:
                stats.best_gap = gap
                if gap > 0:
                    witness = w
                    stats.moment_residual = self.S.moment_residual(w)
                    stats.min_mass = float(np.min(w))
                    break                      # one witness is enough to certify
        return witness, stats


def certified_unreachable(bounds_by_date: dict) -> set:
    """Dates excluded by the LP bounds alone: max_Q W^s < max_{s'} min_Q W^{s'}.

    `bounds_by_date` maps date -> (lower, upper).  If some date s' is guaranteed
    to deliver at least L, any date whose best case is below L cannot be optimal
    for any Q in the identified set.
    """
    if not bounds_by_date:
        return set()
    best_guaranteed = max(lo for lo, _ in bounds_by_date.values())
    return {s for s, (_, hi) in bounds_by_date.items() if hi < best_guaranteed}


def ambiguity_cost(search: "ReachableDateSearch", witnesses: dict) -> list:
    """Cross-member regret among reachable dates (Experiment 6).

    For each ordered pair of reachable dates (s_true, s_committed) with a witness
    Q whose optimum is s_true, the cost of committing to the wrong member is the
    relative regret  1 - W^{s_committed}(Q) / W^{s_true}(Q).
    """
    rows = []
    for s_true, w in witnesses.items():
        if w is None:
            continue
        w_star = search.welfare(w, int(s_true))
        for s_commit in witnesses:
            if int(s_commit) == int(s_true):
                continue
            w_commit = search.welfare(w, int(s_commit))
            rows.append({
                "date_true": int(s_true),
                "date_committed": int(s_commit),
                "welfare_true": w_star,
                "welfare_committed": w_commit,
                "relative_regret": 1.0 - w_commit / w_star if w_star > 0 else np.nan,
            })
    return rows
