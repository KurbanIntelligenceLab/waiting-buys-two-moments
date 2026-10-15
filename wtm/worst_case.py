"""Worst-case rules built on the identified set (draft Section 6).

Both rules need the same expensive object: LP bounds L(s) = min_Q W^s(Q) and
U(s) = max_Q W^s(Q) over M_t built from the planner's *estimated* moments.  They
differ only in the cheap comparison logic applied afterwards.

A DECISION AMBIGUITY, flagged rather than resolved
--------------------------------------------------
The draft describes the two rules with logic that runs in opposite directions,
yet Table 2 assigns both the same guarantee ("never acts too late").

  Envelope (Section 6): "if the achieved value now exceeds the full-information
  value next period, acting now dominates every later date."  That is a *stop
  when dominance is proved* rule.  But proving W^s >= W^{s'} for all s' > s
  implies t* <= s, so such a rule acts at or **after** t* -- it can act too late,
  never too early.

  Conservative-early reading: continue past s only while some later date is
  *provably* better (L(s') > U(s) for some s' > s).  Every date it passes is
  provably not the argmax, so on stopping at s we have t* >= s -- it acts at or
  **before** t*, i.e. "never acts too late", matching Table 2 and matching
  Section 2's finding that worst-case rules are "actively harmful" (they stop
  early and forfeit the information gain).

The second reading is the one consistent with Table 2, Table 2's caption
("soundness here constrains the direction of the error, not its size") and
Section 2.  The first is the more literal reading of the envelope sentence.
**FROZEN 2026-08-21: conservative-early, for both worst-case rules** (D8b).
It is the only reading consistent with Table 2's stated guarantee, and the pilot
confirms it holds on 100% of runs (conservative-late: 62%).  The contradictory
Section 6 envelope sentence is flagged for revision in PAPER_RESULTS_NOTES.md.
The other logic remains implemented so the choice stays auditable; every result
records which was used.

Moment estimation (D8c -> D8d, frozen 2026-08-27).  Two constructions exist:

  `projection`  -- covariance-weighted projection onto the moment cone, then an
                   equality-constrained `M_t`.  Retained for comparison.  It
                   restores feasibility but lands on the cone boundary, where the
                   representing measure is unique, so `M_t` collapses to a point
                   and the rules degenerate into plug-ins.
  `confidence`  -- **the frozen default.**  `M_t^alpha` is the set of grid
                   distributions whose moments lie in a covariance-weighted
                   confidence region around `mhat` (`wtm/confidence_set.py`).
                   Non-empty without projection, has interior by construction,
                   and contains the truth with the region's nominal coverage,
                   which is what restores "never acts too late" in finite
                   samples.

NPMLE-implied moments are deliberately NOT used under either construction: that
would couple these controls to the plug-in family they exist to control for.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from scipy.optimize import linprog

from .confidence_set import ALPHA_DEFAULT, build_region, contains
from .lp import SOLVER, IdentifiedSet, support_grid
from .moment_projection import project_to_cone

FROZEN_SET_MODE = "confidence"       # D8d


@dataclass
class WorstCaseBounds:
    dates: np.ndarray
    lower: dict
    upper: dict
    feasible: bool
    n_lp_solved: int
    runtime_s: float
    status: str = "ok"
    n_infeasible_dates: int = 0
    projection: object = None
    region: object = None
    set_mode: str = FROZEN_SET_MODE


def compute_bounds(moments, t: int, gamma: int, T: int, beta: float,
                   grid_size: int = 1001, scan: dict = None,
                   cov: np.ndarray = None, weighting: str = "covariance",
                   set_mode: str = FROZEN_SET_MODE, alpha: float = ALPHA_DEFAULT,
                   lazy_upper: bool = False) -> WorstCaseBounds:
    """LP bounds on W^s over the identified set, for every candidate date.

    `set_mode="confidence"` (frozen) builds M_t^alpha from a covariance-weighted
    confidence region; `set_mode="projection"` uses the cone-projected point
    estimate.  Diagnostics for whichever was used are attached to the result.
    """
    scan = scan or {}
    t0 = time.perf_counter()
    grid = support_grid(grid_size)
    proj = region = None
    if set_mode == "confidence":
        if cov is None:
            raise ValueError("confidence mode needs the moment covariance")
        region = build_region(moments, cov, t, alpha=alpha)
        S = IdentifiedSet.from_confidence_region(region, t, grid)
    else:
        if cov is not None:
            proj = project_to_cone(moments, cov, grid, t, weighting=weighting)
            moments = proj.moments
        S = IdentifiedSet.from_moments(moments, t, grid)
    w, _msg = S.feasible_point()
    dates = np.arange(1, T)
    if w is None:
        return WorstCaseBounds(dates, {}, {}, False, 1,
                               time.perf_counter() - t0,
                               status="identified set empty",
                               projection=proj, region=region, set_mode=set_mode)
    lower, upper, n_lp, n_bad = {}, {}, 1, 0

    if lazy_upper:
        # conservative-early needs every LOWER bound, but only the UPPER bounds
        # at dates it actually reaches: at date s it asks whether any later date
        # is provably better, i.e. L(s') > U(s) for some s' > s.  It stops at
        # date 1.97 on average out of 11-19 candidates, so ~87% of the upper
        # bounds computed eagerly are never consulted.  Computing them on demand
        # is a pure saving: the returned decision is identical.
        for s in dates:
            s = int(s)
            r = S.min_welfare(s, gamma, T, beta, seed_lambda=None,
                              **{k: v for k, v in scan.items()
                                 if k in ("n_lambda", "n_refine", "n_rounds")})
            n_lp += r[1]
            lower[s] = r[0][0]
            if not np.isfinite(lower[s]):
                n_bad += 1
        for s in dates:
            s = int(s)
            r = S.max_welfare(s, gamma, T, beta, seed_profile=None,
                              **{k: v for k, v in scan.items()
                                 if k in ("n_fill", "n_refine", "n_rounds")})
            n_lp += r[1]
            upper[s] = r[0][0]
            if not np.isfinite(upper[s]):
                n_bad += 1
            if not any(lower[int(s2)] > upper[s] for s2 in dates if int(s2) > s):
                break                      # the walk stops here; no more needed
    else:
        for s in dates:
            s = int(s)
            res = S.bounds(s, gamma, T, beta, truth_profile=None,
                           truth_lambda=None, **scan)
            n_lp += res.n_lp_solved
            if not np.isfinite(res.lower) or not np.isfinite(res.upper):
                n_bad += 1
            lower[s], upper[s] = res.lower, res.upper
    return WorstCaseBounds(dates, lower, upper, True, n_lp,
                           time.perf_counter() - t0, n_infeasible_dates=n_bad,
                           projection=proj, region=region, set_mode=set_mode)


FROZEN_LOGIC = "conservative_early"      # D8b


def decide(b: WorstCaseBounds, logic: str = FROZEN_LOGIC) -> dict:
    """Turn bounds into a date.  `logic` selects between the two readings."""
    if not b.feasible:
        return {"selected_date": -1, "success": False,
                "failure_reason": b.status, "logic": logic}
    dates = [int(s) for s in b.dates]

    if logic == "conservative_early":
        # continue past s only while some later date is PROVABLY better
        for s in dates:
            if s not in b.upper:
                break              # lazy: the walk never reached this date
            later_better = [s2 for s2 in dates
                            if s2 > s and b.lower[s2] > b.upper[s]]
            if not later_better:
                return {"selected_date": s, "success": True,
                        "failure_reason": "", "logic": logic,
                        "n_dates_passed": dates.index(s)}
        return {"selected_date": dates[-1], "success": True,
                "failure_reason": "", "logic": logic,
                "n_dates_passed": len(dates) - 1}

    if logic == "conservative_late":
        # stop at the first s whose worst case dominates every later best case
        for s in dates:
            later = [b.upper[s2] for s2 in dates if s2 > s]
            if not later or b.lower[s] >= max(later):
                return {"selected_date": s, "success": True,
                        "failure_reason": "", "logic": logic,
                        "n_dates_passed": dates.index(s)}
        return {"selected_date": dates[-1], "success": True,
                "failure_reason": "no date dominates; fell through to last",
                "logic": logic, "n_dates_passed": len(dates) - 1}

    raise ValueError(f"unknown logic {logic!r}")


def comparison_summary(b: WorstCaseBounds) -> dict:
    """Diagnostics on how often the bounds actually separate dates."""
    if not b.feasible:
        return {"n_separating_pairs": 0, "n_pairs": 0, "separating_fraction": np.nan,
                "median_overlap": np.nan}
    dates = [int(s) for s in b.dates]
    pairs = [(a, c) for a in dates for c in dates if c > a]
    sep = sum(1 for a, c in pairs
              if b.lower[c] > b.upper[a] or b.lower[a] > b.upper[c])
    overlaps = []
    for a, c in pairs:
        lo = max(b.lower[a], b.lower[c])
        hi = min(b.upper[a], b.upper[c])
        overlaps.append(max(0.0, hi - lo))
    return {"n_separating_pairs": sep, "n_pairs": len(pairs),
            "separating_fraction": sep / len(pairs) if pairs else np.nan,
            "median_overlap": float(np.median(overlaps)) if overlaps else np.nan}


# -- full-information envelope (the second worst-case rule) -------------------
def _full_info_vectors(grid, s: int, T: int):
    """(mass, value) per grid point for the full-information allocation at s."""
    from .observation import utility_poly
    surv = (1.0 - grid) ** (s - 1)
    return surv, surv * utility_poly(s, T).evaluate(grid)


def max_full_info(S, s: int, T: int, beta: float, n_coarse: int = 41,
                  n_refine: int = 9, n_rounds: int = 3):
    """Inner approximation of max_Q Wbar^s(Q) over the identified set.

    Wbar^s ranks by the true p, so the allocation is a threshold: take grid
    points from the top down until the budget is spent.  As with W^s this is a
    nested one-parameter family -- theta counts how many grid points from the
    top are treated, with a fractional last one -- so the same scan applies, and
    for a fixed theta the problem is an ordinary LP in the weights.
    """
    mass, value = _full_info_vectors(S.grid, s, T)
    order = np.argsort(-S.grid)                # highest risk first
    best, n_lp = -np.inf, 0

    def evaluate(theta):
        nonlocal best, n_lp
        whole = int(np.floor(theta))
        frac = theta - whole
        take = np.zeros(len(S.grid))
        if whole > 0:
            take[order[:whole]] = 1.0
        if whole < len(S.grid):
            take[order[whole]] = frac
        c, bud = take * value, take * mass
        Au, bu = S._ub(bud[None, :], [beta])
        res = linprog(c=-c, A_ub=Au, b_ub=bu, A_eq=S.A, b_eq=S.b,
                      bounds=[(0.0, None)] * S.n, method=SOLVER)
        n_lp += 1
        if res.success and -res.fun > best:
            best = float(-res.fun)
            return theta
        return None

    # bracket: treating the top j grid points must be affordable for some Q
    hi = float(len(S.grid))
    for j in range(1, len(S.grid) + 1):
        m = np.zeros(len(S.grid))
        m[order[:j]] = mass[order[:j]]
        Au, bu = S._ub()
        res = linprog(c=m, A_ub=Au, b_ub=bu, A_eq=S.A, b_eq=S.b,
                      bounds=[(0.0, None)] * S.n, method=SOLVER)
        n_lp += 1
        if not res.success:
            break
        if res.fun > beta + 1e-15:
            hi = float(j)
            break

    thetas = np.linspace(0.0, hi, int(n_coarse))
    argbest = 0.0
    for th in thetas:
        got = evaluate(float(th))
        if got is not None:
            argbest = got
    step = hi / max(int(n_coarse) - 1, 1)
    for _ in range(int(n_rounds)):
        lo_, hi_ = max(0.0, argbest - step), min(hi, argbest + step)
        for th in np.linspace(lo_, hi_, int(n_refine)):
            got = evaluate(float(th))
            if got is not None:
                argbest = got
        step /= (int(n_refine) - 1) / 2.0
    return best, n_lp


def decide_envelope(S, dates, lower, gamma: int, T: int, beta: float,
                    scan: dict = None):
    """Worst-case ENVELOPE rule, conservative-early (D8b).

    Same comparison as the LP rule, but the upper bound on later welfare comes
    from the full-information envelope: Wbar is non-increasing and dominates W
    (Lemma 2), so Wbar^{s+1} caps every later W^{s'}.  Using U(Wbar^{s+1}) in
    place of U(W^s) is a strictly looser relaxation -- it can only make the
    "waiting is provably better" test harder -- which is what Table 2 means by
    "the envelope relaxation may be loose".

    Takes the already-computed lower bounds so the two worst-case rules share
    that work instead of duplicating it.
    """
    scan = scan or {}
    n_lp = 0
    dates = [int(x) for x in dates]
    for s in dates:
        if s == dates[-1]:
            return {"selected_date": s, "success": True, "failure_reason": "",
                    "logic": "conservative_early_envelope", "n_lp_solved": n_lp}
        u_env, k = max_full_info(S, s + 1, T, beta, **scan)
        n_lp += k
        if not any(lower[s2] > u_env for s2 in dates if s2 > s):
            return {"selected_date": s, "success": True, "failure_reason": "",
                    "logic": "conservative_early_envelope", "n_lp_solved": n_lp}
    return {"selected_date": dates[-1], "success": True, "failure_reason": "",
            "logic": "conservative_early_envelope", "n_lp_solved": n_lp}
