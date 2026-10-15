"""Christensen, Moon & Schorfheide (2026) bootstrap decision rule.

Review of Economic Studies, doi:10.1093/restud/rdag017; preprint arXiv:2204.11748.

THE RULE (their Section 3.4). With `mu` the point-identified reduced-form
parameter, `Theta_0(mu)` the known set-valued map giving the identified set for
the payoff-relevant parameter, and `r(d, theta)` the risk of action `d`,

    R(d, mu)   = sup_{theta in Theta_0(mu)} r(d, theta)          (their eq. 9)
    R*_n(d)    = E*_n[ R(d, muhat*) ]                            (bootstrap average)
    delta**_n  = argmin_d R*_n(d).

Their footnote 6 states the loss nests welfare regret explicitly, by taking
`l(d, y, theta, mu) = max_d' E_theta[W(d', ...)] - W(d, y, ...)`, and they do
not distinguish risk from regret thereafter.

THE MAPPING HERE.

    mu            the moments m_1..m_{2 tau - 1} that the length-tau prefix
                  point identifies (Lemma 1)
    Theta_0(mu)   M_tau(mu), the distributions on [0,1] whose moments are mu --
                  the EXACT moment-matching set, not a confidence region
    theta         the latent risk distribution Q
    D             the allocation dates 1..T-1
    r(d, Q)       regret of date d under Q

WHY THE FEASIBLE SET IS NOT THE ONE USED BY `wtm/minimax_regret.py`. That rule
builds the identified set from a covariance-weighted CONFIDENCE REGION around
the point estimate, which folds sampling uncertainty into the ambiguity set.
CMS deliberately separate the two: ambiguity is handled by the exact
moment-matching set conditional on each mu, and sampling uncertainty by
averaging over the bootstrap distribution of muhat. So `Theta_0(muhat*)` here
is `IdentifiedSet.from_moments`, evaluated at each bootstrap draw's moments.

PROJECTION ONTO THE CONE. A bootstrap draw's raw moment vector need not be
attainable by any distribution on [0,1], in which case `M_tau(muhat*)` is empty
and `R(d, muhat*)` is undefined. We project each draw onto the truncated moment
cone with the pipeline's existing covariance-weighted projection before forming
the set. That is a modelling choice on our side, not theirs, and it is recorded
per draw so its frequency can be reported.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .lp import IdentifiedSet, support_grid
from .minimax_regret import _pair_sup, _theta_bracket
from .lp import group_matrices


@dataclass
class CMSResult:
    dates: np.ndarray
    avg_max_risk: dict                 # d -> R*_n(d)
    selected_date: int
    n_draws_used: int
    n_draws_requested: int
    n_projected: int
    n_infeasible: int
    n_lp_solved: int
    runtime_s: float
    n_exact: int = 0                # draws scored exactly (genuine singleton)
    per_draw: list = field(default_factory=list)   # [{d: R(d, muhat*_b)}]
    status: str = ""


def _pair_sup_fast(solver, d: int, r: int, hi_theta: float,
                   n_lambda: int, n_theta: int, n_refine: int, n_rounds: int):
    """Same coarse-plus-refine search as `minimax_regret._pair_sup`, on the
    persistent-model solver. Identical scan points, so the estimate is the same
    quantity; only the LP backend differs."""
    best = (-np.inf, 0.0, 0.0)

    def ev(theta, lam):
        nonlocal best
        v = solver.value(r, float(theta), float(lam), d)
        if v > best[0]:
            best = (v, float(theta), float(lam))

    for th in np.linspace(0.0, hi_theta, n_theta):
        for lam in np.linspace(0.0, 1.0, n_lambda):
            ev(th, lam)
    d_th = hi_theta / max(n_theta - 1, 1)
    d_lam = 1.0 / max(n_lambda - 1, 1)
    for _ in range(int(n_rounds)):
        _, th0, lam0 = best
        for th in np.linspace(max(0.0, th0 - d_th), min(hi_theta, th0 + d_th), n_refine):
            for lam in np.linspace(max(0.0, lam0 - d_lam), min(1.0, lam0 + d_lam), n_refine):
                ev(th, lam)
        d_th /= (n_refine - 1) / 2.0
        d_lam /= (n_refine - 1) / 2.0
    return best[0]


def max_risk(S: IdentifiedSet, gamma: int, T: int, beta: float,
             n_lambda: int = 11, n_theta: int = 7, n_refine: int = 5,
             n_rounds: int = 3, fast: bool = True, prune: bool = False):
    """R(d, mu) = sup_{Q in S} [ max_r W^r(Q) - W^d(Q) ] for every date d.

    The sup and the max over r commute because the date set is finite, so the
    pairwise programs are solved one at a time, exactly as in the minimax-regret
    rule. The estimate is one-signed: every scanned (theta, lambda) is feasible,
    so each value lands at or below the true sup.
    """
    dates = np.arange(1, T)
    mats = {int(s): group_matrices(S.grid, int(s), gamma, T) for s in dates}
    n_lp = 0
    brackets = {}
    for s in dates:
        hi, k = _theta_bracket(S, mats[int(s)][0], int(s), beta)
        brackets[int(s)] = hi
        n_lp += k

    # Floor every date at the regret of a FEASIBLE point of the set rather than
    # at zero. Any Q in Theta_0 gives a valid lower bound on the sup, so taking
    # the larger of the two keeps the estimate one-signed while removing the
    # false ties that flooring at zero creates: at a single Q exactly one date
    # has zero regret, so at most one date can still tie at zero.
    w0 = S.feasible_point()[0]
    n_lp += 1
    floor = (exact_regret_at_point(w0, S.grid, gamma, T, beta)
             if w0 is not None else {int(d): 0.0 for d in dates})

    # Lossless pruning: a pair whose valid upper bound sits at or below the
    # running incumbent cannot raise R(d), so its scan is skipped. R(d) is
    # unchanged exactly. See wtm/prune.py.
    ub = {}
    if prune:
        try:
            from .prune import pair_upper_bounds
            ub, k = pair_upper_bounds(S, mats, dates, brackets, beta)
            n_lp += k
        except Exception:                                      # noqa: BLE001
            ub = {}

    solver = None
    if fast:
        try:
            from .fastlp import PairSolver
            solver = PairSolver(S, mats, gamma, T, beta)
        except Exception:                                      # noqa: BLE001
            solver = None                                      # fall back below

    out = {}
    for d in [int(x) for x in dates]:
        PHI_d, UTIL_d = mats[d][0], mats[d][1]
        worst = float(floor.get(d, 0.0))               # r = d gives regret 0
        # strongest candidates first, so the incumbent rises fast and prunes more
        others = [int(x) for x in dates if int(x) != d]
        if ub:
            others.sort(key=lambda r: -ub.get((r, d), np.inf))
        for r in others:
            # guard the comparison: the bound holds exactly, but both sides
            # carry solver error of order 1e-9 relative, and an unguarded test
            # was measured skipping pairs that mattered (by 4e-5 relative).
            if ub and ub.get((r, d), np.inf) <= worst * (1.0 - 1e-6) - 1e-15:
                continue                                   # cannot raise R(d)
            if solver is not None:
                v = _pair_sup_fast(solver, d, r, brackets[r], n_lambda,
                                   n_theta, n_refine, n_rounds)
            else:
                PHI_r, UTIL_r = mats[r][0], mats[r][1]
                v, k = _pair_sup(S, UTIL_r, PHI_r, UTIL_d, PHI_d, r,
                                 brackets[r], beta, n_lambda, n_theta,
                                 n_refine, n_rounds)
                n_lp += k
            if v > worst:
                worst = v
        out[d] = worst
    if solver is not None:
        n_lp += solver.n_solved
    return out, n_lp


def prefailure_counts(counts, tau: int) -> np.ndarray:
    """Merge each (exit at the end of period tau, history) path cell into the
    (still active, same history) cell.

    The two cells have identical columns in the moment map, but a planner
    observing a length-tau prefix cannot tell them apart: the period-tau exit
    is visible only at tau+1. After merging, the counts are a function of the
    pre-failure prefix alone.
    """
    from .moment_cone import build_path_alphabet
    pa = build_path_alphabet(tau)
    idx = {p: i for i, p in enumerate(pa.paths)}
    c = np.array(counts, dtype=float)
    if c.shape != (len(pa.paths),):
        raise ValueError(f"expected {len(pa.paths)} path counts, got {c.shape}")
    for (e, obs), i in idx.items():
        if e == tau:
            c[idx[(None, obs)]] += c[i]
            c[i] = 0.0
    return c


RESAMPLING = ("prefailure", "paths")

# Bootstrap random-number streams.
#   "legacy"       the archived Table 3 scheme: draw d of seed index s uses
#                  default_rng((base + s + d) mod 2^32). That reuses the data
#                  seed for draw 0 and other runs' data seeds for later draws,
#                  contrary to the frozen seed protocol (config: seeds.
#                  seed_protocol.rule_seed); kept only to reproduce the archive.
#   "independent"  the frozen protocol's rule stream, split per draw:
#                  SeedSequence(entropy=rule_seed, spawn_key=(CMS_STREAM_TAG, d))
#                  with rule_seed = (sim_seed + 2^31 - 1) mod 2^32. Child
#                  sequences with distinct spawn keys are independent of each
#                  other and of every integer-seeded stream (data or rule).
RNG_SCHEMES = ("legacy", "independent")
CMS_STREAM_TAG = 0x434D53          # "CMS"


def draw_seed(scheme: str, base: int, seed_index: int, draw: int):
    """Seed for bootstrap draw `draw` of run (cell base, seed_index)."""
    if scheme == "legacy":
        return (int(base) + int(seed_index) + int(draw)) % 2 ** 32
    if scheme == "independent":
        sim_seed = (int(base) + int(seed_index)) % 2 ** 32
        rule_seed = (sim_seed + 2 ** 31 - 1) % 2 ** 32
        return np.random.SeedSequence(entropy=rule_seed,
                                      spawn_key=(CMS_STREAM_TAG, int(draw)))
    raise ValueError(f"rng scheme must be one of {RNG_SCHEMES}")


@dataclass
class DrawOutcome:
    risks: dict | None          # {d: R(d, muhat*_b)}, None if the draw is unusable
    projected: bool
    exact: bool
    n_lp: int


def draw_risks(draw, tau: int, gamma: int, T: int, beta: float, grid,
               scan: dict = None, exact_on_singleton: bool = True) -> DrawOutcome:
    """R(d, muhat*_b) for one bootstrap draw of path counts.

    This is the only place a draw enters the rule, and it enters only through
    `moments_with_covariance(draw)`, i.e. through (mhat, Cov(mhat)). Every
    later step -- the identified set, the feasibility check, the projection,
    the singleton test and the inner sup -- is a deterministic function of
    those two arrays and the fixed settings. Because the moment map gives the
    same column to a period-tau exit cell and to the matching active cell,
    (mhat, Cov) are, in exact arithmetic, functions of the merged (pre-failure)
    counts; see paper/reconciliation/README.md section 06.
    """
    from .moment_projection import moments_with_covariance, project_to_cone

    scan = scan or {}
    mhat, cov = moments_with_covariance(draw, tau, gamma)
    mhat_used = mhat
    projected = False
    S = IdentifiedSet.from_moments(mhat, tau, grid)
    if S.feasible_point()[0] is None:
        proj = project_to_cone(mhat, cov, grid, tau)
        mhat_used = proj.moments
        S = IdentifiedSet.from_moments(mhat_used, tau, grid)
        projected = True
        if S.feasible_point()[0] is None:
            return DrawOutcome(None, projected, False, 0)
    # A genuine singleton is decided by the CONTINUUM test, not by the grid
    # set's width: at grid 201 the grid set collapses to a vertex in 62% of
    # draws where the continuum set is not a point at all, so the grid width
    # is not evidence of uniqueness. See results/cms_compatibility_assessment.md.
    is_point = False
    if exact_on_singleton:
        is_point = on_cone_boundary(mhat_used, tau, tol=1e-9)[0]
    if is_point:
        w = S.feasible_point()[0]
        return DrawOutcome(exact_regret_at_point(w, S.grid, gamma, T, beta),
                           projected, True, 0)
    risks, k = max_risk(S, gamma, T, beta, **scan)
    return DrawOutcome(risks, projected, False, k)


def bootstrap_rule(counts, tau: int, gamma: int, T: int, beta: float,
                   n_boot: int = 20, grid_size: int = 401, seed=0,
                   scan: dict = None, exact_on_singleton: bool = True,
                   resampling: str = "paths") -> CMSResult:
    """The bootstrap decision, from observed path counts of the length-tau prefix.

    The bootstrap resamples counts multinomially, which is the nonparametric
    bootstrap for this multinomial experiment. `seed` is an integer or a
    numpy SeedSequence (see `draw_seed` for the two archived/future schemes).

    resampling="paths" (default) resamples the raw path cells. This is the
    manuscript's implementation: it produced the Table 3 CMS row
    (results/cms_table2_raw.csv).

    resampling="prefailure" is an alternative, proposed for the authors'
    decision only (paper/reconciliation/README.md section 06). It merges the
    period-tau exit cells into the matching active cells first
    (`prefailure_counts`), so every realized draw is a function of the
    pre-failure prefix and the seed.

    IDEALIZED EQUIVALENCE. A draw enters the rule only through `draw_risks`,
    i.e. through moments_with_covariance, whose map gives an exit cell and its
    active cell the same column, so in exact arithmetic the rule depends on a
    draw only through its merged counts. Merging a Multinomial(n, p) draw gives
    Multinomial(n, merged p). Hence, if the bootstrap randomness is independent
    of the data, "resample paths, then merge" and "merge, then resample" give
    the recommendation the same conditional law. This is a statement about the
    idealized randomized procedure. The archived run used the "legacy" seeds,
    which reuse data seeds (see `draw_seed`), so for it the statement is a
    modelling assumption, checked empirically (results/audit/cms_seed_audit.json),
    not a theorem. In floating point the two summation orders agree only up to
    rounding; results/audit/cms_equivalence_audit.json gives empirical checks,
    not a uniform bound.

    They are not seed-identical: numpy's multinomial draws category by
    category with conditional binomials, so merging two categories changes
    which random numbers every later category consumes. For a fixed seed the
    two modes give different draws (results/audit/cms_window_audit.json),
    without either using period-tau exits.
    See results/audit/cms_equivalence_audit.json and
    paper/reconciliation/README.md section 06.
    """
    if resampling not in RESAMPLING:
        raise ValueError(f"resampling must be one of {RESAMPLING}")
    t0 = time.perf_counter()
    scan = scan or {}
    grid = support_grid(grid_size)
    counts = np.asarray(counts, dtype=float)
    if resampling == "prefailure":
        counts = prefailure_counts(counts, tau)
    n_total = int(round(counts.sum()))
    rng = np.random.default_rng(seed)

    dates = np.arange(1, T)
    totals = {int(d): 0.0 for d in dates}
    per_draw, n_lp = [], 0
    used = projected = infeasible = n_exact = 0

    for _ in range(int(n_boot)):
        draw = rng.multinomial(n_total, counts / counts.sum()).astype(float)
        out = draw_risks(draw, tau, gamma, T, beta, grid, scan,
                         exact_on_singleton)
        projected += out.projected
        if out.risks is None:
            infeasible += 1
            continue
        n_lp += out.n_lp
        n_exact += out.exact
        per_draw.append(out.risks)
        for d, v in out.risks.items():
            totals[d] += v
        used += 1

    if used == 0:
        return CMSResult(dates, {}, -1, 0, int(n_boot), projected, infeasible,
                         n_lp, time.perf_counter() - t0,
                         status="no usable bootstrap draw")
    avg = {d: totals[d] / used for d in totals}
    sel = min(avg, key=lambda d: (avg[d], d))        # earliest date on ties
    return CMSResult(dates, avg, int(sel), used, int(n_boot), projected,
                     infeasible, n_lp, time.perf_counter() - t0,
                     n_exact=n_exact, per_draw=per_draw)


# --------------------------------------------------------------------------
# Singleton draws: an exact path, and two separate tests for detecting them
# --------------------------------------------------------------------------

def hankel_forms(moments, tau: int):
    """The two truncated-Hausdorff Hankel forms for m_0..m_{2 tau - 1}.

    For an odd-order sequence m_0..m_{2k+1}, a measure on [0,1] exists iff both
    A = [m_{i+j+1}] and B = [m_{i+j} - m_{i+j+1}] (i, j = 0..k) are PSD. Here
    2 tau - 1 = 2k + 1, so k = tau - 1 and both forms are tau x tau.

    This is a statement about the CONTINUUM moment problem. It is independent of
    any support grid, which the LP-based width test is not.
    """
    m = np.asarray(moments, dtype=float)
    if m.shape[0] < 2 * tau:
        raise ValueError(f"need m_0..m_{2*tau-1}, got {m.shape[0]} entries")
    k = tau - 1
    A = np.array([[m[i + j + 1] for j in range(k + 1)] for i in range(k + 1)])
    B = np.array([[m[i + j] - m[i + j + 1] for j in range(k + 1)]
                  for i in range(k + 1)])
    return A, B


def on_cone_boundary(moments, tau: int, tol: float = 1e-12):
    """Is the moment vector on the boundary of the truncated moment cone?

    Returns (is_boundary, min_eigenvalue, rank_deficiency). A moment sequence in
    the INTERIOR has both Hankel forms positive definite and admits many
    representing measures; one on the boundary has a singular form, and the
    representing measure is then determined (it is the finitely atomic measure
    supported on the roots associated with the kernel).

    The caller should scale `tol` to the moments' magnitude; these moments can be
    of order 1e-3 or smaller, so an absolute tolerance is meaningless. We
    normalise each form by its largest eigenvalue.
    """
    A, B = hankel_forms(moments, tau)
    out = []
    for M in (A, B):
        w = np.linalg.eigvalsh((M + M.T) / 2.0)
        top = max(abs(w).max(), 1e-300)
        out.append((w.min() / top, int((np.abs(w) / top < tol).sum())))
    min_rel = min(o[0] for o in out)
    defic = max(o[1] for o in out)
    return bool(min_rel < tol), float(min_rel), int(defic)


def welfare_trajectory(w, grid, gamma: int, T: int, beta: float):
    """W^s for s = 1..T-1 under the single distribution `w` on `grid`.

    Exact given `w`: the inner allocation is the greedy fill of the knapsack,
    which is the LP's exact solution because the constraint matrix is an
    interval matrix in the group ordering.
    """
    from .welfare import fill_greedy
    out = np.empty(T - 1, dtype=float)
    for s in range(1, T):
        PHI, UTIL = group_matrices(grid, s, gamma, T)
        val, _ = fill_greedy(PHI @ w, UTIL @ w, beta)
        out[s - 1] = val
    return out


def exact_regret_at_point(w, grid, gamma: int, T: int, beta: float):
    """R(d) for every date when Theta_0 is the single distribution `w`.

    No scan: with one distribution in the set the sup is attained there, so
    R(d) = max_r W^r - W^d exactly. This both removes the scan's cost on such
    draws and removes the false ties its under-estimation creates.
    """
    W = welfare_trajectory(w, grid, gamma, T, beta)
    best = float(W.max())
    return {d + 1: float(best - W[d]) for d in range(len(W))}


def grid_set_width(S: IdentifiedSet, power: int):
    """Exact range of the linear functional E[p^power] over S.

    This is a property of the GRID-RESTRICTED set. A zero width here says the
    grid set is a point; it does not by itself establish that the continuum
    identified set is a point. Use `on_cone_boundary` for that.
    """
    from scipy.optimize import linprog
    from .lp import SOLVER
    c = S.grid ** power
    Au, bu = S._ub()
    lo = linprog(c=c, A_ub=Au, b_ub=bu, A_eq=S.A, b_eq=S.b,
                 bounds=[(0.0, None)] * S.n, method=SOLVER)
    hi = linprog(c=-c, A_ub=Au, b_ub=bu, A_eq=S.A, b_eq=S.b,
                 bounds=[(0.0, None)] * S.n, method=SOLVER)
    if not (lo.success and hi.success):
        return None
    return float(-hi.fun) - float(lo.fun)
