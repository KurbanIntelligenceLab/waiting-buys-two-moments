"""Minimax-regret allocation over the estimated identified set.

WHAT THIS IS. For a candidate date s the maximum regret over the set is

    R(s) = max_{r}  sup_{Q in M}  [ W^r(Q) - W^s(Q) ],

and the rule selects argmin_s R(s). The outer max and the inner sup commute
because the date set is finite, so the pairwise programs can be solved one at
a time -- this is the joint pairwise program, not the separate-bounds test the
two conservative rules use.

WHY IT IS AN LP SCAN AND NOT A BILINEAR PROGRAM. Write the later date's welfare
in its primal form and the current date's in its dual form:

    W^r(Q) = max_theta f_r(theta, Q),
    W^s(Q) = min_lambda g_s(lambda, Q),   g_s = beta*lambda + sum_j (v_j - lambda m_j)_+

so that

    sup_Q [W^r(Q) - W^s(Q)] = max_theta max_lambda sup_Q [ f_r(theta,Q) - g_s(lambda,Q) ].

For FIXED (theta, lambda) the inner problem is an ordinary LP in (w, mu): the
positive parts linearise exactly the way they do in `IdentifiedSet.min_welfare`,
because the mu block is minimised inside a maximisation.

THE APPROXIMATION IS ONE-SIDED, which is the point of this construction. Any
feasible theta gives f_r <= W^r, and any lambda gives g_s >= W^s, so every
scanned pair yields a value at or below sup_Q [W^r - W^s]. The scan therefore
UNDER-estimates each date's maximum regret, at every date, in the same
direction. It is still an approximation and no certificate is claimed; but
unlike the separate-bounds dominance test -- where the two errors enter with
opposite signs -- refining this scan moves every date's score the same way, so
the argmin is not driven by a difference of two opposing errors.

COST. The scan is over (theta, lambda) per ordered pair of dates. The theta
bracket from `max_welfare` applies unchanged: at these budget fractions only
the top few y-groups are ever affordable, so theta ranges over about a dozen
values rather than s+1.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linprog

from .lp import SOLVER, group_matrices, theta_to_cutoff


@dataclass
class RegretResult:
    dates: np.ndarray
    max_regret: dict                      # s -> lower estimate of R(s)
    selected_date: int
    feasible: bool
    n_lp_solved: int
    runtime_s: float
    status: str = ""
    argmax_r: dict = field(default_factory=dict)


def _theta_bracket(S, PHI, s: int, beta: float):
    """The affordable range of theta at date s, as in IdentifiedSet.max_welfare.

    Returns (hi_theta, n_lp). Never narrows on a solver failure.
    """
    hi = float(s + 1)
    n_lp = 0
    for j in range(1, s + 2):
        mass = PHI[s - j + 1:].sum(axis=0)
        Au, bu = S._ub()
        res = linprog(c=mass, A_ub=Au, b_ub=bu, A_eq=S.A, b_eq=S.b,
                      bounds=[(0.0, None)] * S.n, method=SOLVER)
        n_lp += 1
        if not res.success:
            break
        if res.fun > beta + 1e-15:
            hi = float(j)
            break
    return hi, n_lp


def _pair_value(S, UTIL_r, PHI_r, UTIL_s, PHI_s, r: int, theta: float,
                lam: float, beta: float):
    """sup_{Q in M} [ f_r(theta, Q) - g_s(lambda, Q) ] for fixed theta, lambda.

    Variables are (w, mu) with mu the positive-part block at date s.
    """
    k, f = theta_to_cutoff(theta, r)
    c_r = UTIL_r[k + 1:].sum(axis=0) + f * UTIL_r[k]
    bud_r = PHI_r[k + 1:].sum(axis=0) + f * PHI_r[k]
    n_s = PHI_s.shape[0]

    # maximise c_r . w - sum(mu)  (the -beta*lambda constant is added after)
    c = np.concatenate([-c_r, np.ones(n_s)])          # linprog minimises
    # mu_j >= (UTIL_s - lam*PHI_s)_j . w   ->   (UTIL_s - lam*PHI_s) w - mu <= 0
    rows = np.hstack([UTIL_s - lam * PHI_s, -np.eye(n_s)])
    extra_A = np.vstack([rows, np.concatenate([bud_r, np.zeros(n_s)])])
    extra_b = np.concatenate([np.zeros(n_s), [beta]])
    A_ub, b_ub = S._ub(extra_A, extra_b, pad=n_s)
    A_eq = np.hstack([S.A, np.zeros((S.A.shape[0], n_s))])
    res = linprog(c=c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=S.b,
                  bounds=[(0.0, None)] * (S.n + n_s), method=SOLVER)
    if not res.success:
        return -np.inf
    return float(-res.fun - beta * lam)


def _pair_sup(S, UTIL_r, PHI_r, UTIL_s, PHI_s, r: int, hi_theta: float,
              beta: float, n_lambda: int, n_theta: int, n_refine: int,
              n_rounds: int):
    """Lower estimate of sup_Q [W^r - W^s], by a coarse scan over (theta,
    lambda) followed by local refinement around the incumbent.

    Refinement matters: a flat lambda grid at half the resolution changes the
    selected date, while halving the theta grid or the support grid does not
    (see results/minimax_regret_report.md). Refining around the incumbent buys
    the lambda resolution at a fraction of the LP count, and keeps the estimate
    one-signed -- every evaluated pair is still a feasible (theta, lambda), so
    the running maximum never exceeds the true sup.
    """
    best = (-np.inf, 0.0, 0.0)
    n_lp = 0

    def ev(theta: float, lam: float):
        nonlocal best, n_lp
        v = _pair_value(S, UTIL_r, PHI_r, UTIL_s, PHI_s, r,
                        float(theta), float(lam), beta)
        n_lp += 1
        if v > best[0]:
            best = (v, float(theta), float(lam))

    thetas = np.linspace(0.0, hi_theta, n_theta)
    lams = np.linspace(0.0, 1.0, n_lambda)
    for th in thetas:
        for lam in lams:
            ev(th, lam)

    d_th = hi_theta / max(n_theta - 1, 1)
    d_lam = 1.0 / max(n_lambda - 1, 1)
    for _ in range(int(n_rounds)):
        _, th0, lam0 = best
        th_lo, th_hi = max(0.0, th0 - d_th), min(hi_theta, th0 + d_th)
        lam_lo, lam_hi = max(0.0, lam0 - d_lam), min(1.0, lam0 + d_lam)
        for th in np.linspace(th_lo, th_hi, n_refine):
            for lam in np.linspace(lam_lo, lam_hi, n_refine):
                ev(th, lam)
        d_th /= (n_refine - 1) / 2.0
        d_lam /= (n_refine - 1) / 2.0
    return best[0], n_lp


def compute(S, gamma: int, T: int, beta: float, n_lambda: int = 11,
            n_theta: int = 7, n_refine: int = 5, n_rounds: int = 3) -> RegretResult:
    """Minimax-regret date over the identified set `S`."""
    t0 = time.perf_counter()
    dates = np.arange(1, T)
    w0, msg = S.feasible_point()
    if w0 is None:
        return RegretResult(dates, {}, -1, False, 1,
                            time.perf_counter() - t0, status="identified set empty")

    mats = {int(s): group_matrices(S.grid, int(s), gamma, T) for s in dates}
    n_lp = 0
    brackets = {}
    for s in dates:
        PHI_s, _ = mats[int(s)]
        hi, k = _theta_bracket(S, PHI_s, int(s), beta)
        brackets[int(s)] = hi
        n_lp += k

    max_regret, argmax_r = {}, {}
    for s in [int(x) for x in dates]:
        PHI_s, UTIL_s = mats[s][0], mats[s][1]
        worst, worst_r = 0.0, s               # r = s gives regret 0
        for r in [int(x) for x in dates]:
            if r == s:
                continue
            PHI_r, UTIL_r = mats[r][0], mats[r][1]
            best, k = _pair_sup(S, UTIL_r, PHI_r, UTIL_s, PHI_s, r,
                                brackets[r], beta, n_lambda, n_theta,
                                n_refine, n_rounds)
            n_lp += k
            if best > worst:
                worst, worst_r = best, r
        max_regret[s] = worst
        argmax_r[s] = worst_r

    sel = min(max_regret, key=lambda s: (max_regret[s], s))   # earliest on ties
    return RegretResult(dates, max_regret, int(sel), True, n_lp,
                        time.perf_counter() - t0, argmax_r=argmax_r)
