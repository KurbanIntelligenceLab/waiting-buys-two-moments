"""Identified sets and LP bounds on future welfare (Experiments 4-6).

The identified set after a prefix of length t is

    M_t = { Q on [0,1] : E_Q[p^j] = m_j,  j = 1..2t-1 }        (plus E_Q[1] = 1)

discretised on a common support grid, so Q is a weight vector w >= 0 on the grid.

Bounding W^s over M_t is not a plain LP, because W^s(Q) is itself the value of
an inner allocation LP.  Two facts make it tractable and, more importantly, make
the reported interval *provably contain the truth*:

  max side.  W^s(Q) = max_a L(a, Q), so
                 max_{Q in M} W^s(Q) = max_a max_{Q in M} L(a, Q).
             For a *fixed* allocation a the inner problem is an ordinary LP.  The
             optimal a is a cutoff rule (MLR), so a is indexed by a cutoff group
             k* and a fill fraction f in [0,1], and we scan (k*, f).  Any (k*, f)
             yields a feasible allocation, hence a valid *lower* estimate of the
             max.  Seeding the scan with the truth's own (k*, f) therefore
             guarantees  max_hat >= W^s(P_true).

  min side.  By LP duality W^s(Q) = min_{lambda>=0} [ beta*lambda +
             sum_k (v_k(Q) - lambda*m_k(Q))_+ ], and min_Q min_lambda commutes.
             For a *fixed* lambda the problem is an ordinary LP in (w, mu).  Any
             lambda yields  h(lambda) >= min_Q W^s(Q), i.e. a valid *upper*
             estimate of the min, and h(lambda_true) <= W^s(P_true), so seeding
             the scan with the truth's own dual price guarantees
             min_hat <= W^s(P_true).

Both scans therefore produce an **inner** approximation of the true identified
interval: the reported width is a lower bound on the true width, and truth
containment holds by construction rather than by luck.  That is the conservative
direction for any claim that identification is poor.

W^s is a pointwise maximum of functions linear in Q and hence convex in Q, so
the max is attained at a vertex of M_t -- a distribution supported on at most 2t
grid points.  That is a useful cross-check on anything the search returns.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linprog

from .observation import phi, utility_poly
from .welfare import fill_greedy

SOLVER = "highs"


def support_grid(n_points: int, kind: str = "uniform") -> np.ndarray:
    """Common support grid on [0, 1]."""
    if kind == "uniform":
        return np.linspace(0.0, 1.0, int(n_points))
    if kind == "chebyshev":
        k = np.arange(int(n_points))
        return 0.5 * (1.0 - np.cos(np.pi * k / (int(n_points) - 1)))
    raise ValueError(f"unknown grid kind {kind!r}")


def moment_constraints(dist, t: int, grid: np.ndarray, scale: bool = True):
    """(A, b) enforcing E_Q[p^j] = E_P[p^j] for j = 0..2t-1 on the grid.

    Returns (A, b, row_scale).  Each row is divided by its own right-hand side,
    turning every constraint into `E_Q[p^j] / m_j = 1`.  That is an exact
    algebraic rescaling -- the feasible set is unchanged -- but it is not
    optional in practice.  For Beta(1,32) at t = 4 the raw right-hand sides run
    from 1 down to 6.5e-08, and HiGHS declares the system *infeasible* rather
    than solving it; every one of those cells then reports no bound at all.
    After scaling, all right-hand sides are 1 and the same systems solve.

    `row_scale` is retained so moment residuals can be reported back in the
    original, interpretable units rather than in scaled ones.
    """
    n_mom = 2 * t - 1
    A = np.vstack([grid ** j for j in range(n_mom + 1)])
    b = np.array([1.0] + [dist.raw_moment(j) for j in range(1, n_mom + 1)])
    row_scale = np.ones_like(b)
    if scale:
        for j in range(len(b)):
            s = abs(b[j]) if abs(b[j]) > 0 else np.max(np.abs(A[j]))
            row_scale[j] = s if s > 0 else 1.0
        A = A / row_scale[:, None]
        b = b / row_scale
    return A, b, row_scale


def group_matrices(grid: np.ndarray, s: int, gamma: int, T: int):
    """PHI[k, i] = phi_{s,k}(grid_i) and UTIL[k, i] = phi_{s,k}(grid_i) u^s(grid_i)."""
    u = utility_poly(s, T).evaluate(grid)
    PHI = np.vstack([phi(s, k, gamma).evaluate(grid) for k in range(s + 1)])
    return PHI, PHI * u[None, :]


def welfare_of_weights(grid, w, s: int, gamma: int, T: int, beta: float) -> float:
    """W^s of the discrete distribution (grid, w)."""
    PHI, UTIL = group_matrices(grid, s, gamma, T)
    return fill_greedy(PHI @ w, UTIL @ w, beta)[0]


def _profile_of_weights(grid, w, s, gamma, T, beta):
    PHI, UTIL = group_matrices(grid, s, gamma, T)
    m, v = PHI @ w, UTIL @ w
    _, a = fill_greedy(m, v, beta)
    return m, v, a


def theta_to_cutoff(theta: float, s: int):
    """Map the nested allocation parameter theta in [0, s+1] to (cutoff, fill).

    theta counts how much of the active pool is treated, measured from the top
    y-group downwards: theta = 0 treats nobody, theta = 1 treats all of group s,
    theta = s+1 treats the whole active pool.
    """
    theta = float(np.clip(theta, 0.0, s + 1))
    whole = int(np.floor(theta))
    frac = theta - whole
    if whole >= s + 1:
        return 0, 1.0
    return s - whole, frac



def cutoff_and_fill(a: np.ndarray):
    """Recover (cutoff group k*, fill fraction f) from a greedy profile."""
    frac = np.where((a > 1e-12) & (a < 1.0 - 1e-12))[0]
    if len(frac):
        k = int(frac[0])
        return k, float(a[k])
    on = np.where(a > 1e-12)[0]
    if len(on) == 0:
        return len(a) - 1, 0.0
    return int(on.min()), 1.0


@dataclass
class LPResult:
    lower: float
    upper: float
    lower_status: str
    upper_status: str
    lower_residual: float
    upper_residual: float
    n_lp_solved: int
    lambda_star: float = np.nan
    cutoff_star: int = -1
    fill_star: float = np.nan
    min_weight: float = np.nan
    notes: str = ""


class IdentifiedSet:
    """M_t on a fixed grid, with welfare bounds for any candidate date s."""

    def __init__(self, dist, t: int, grid: np.ndarray):
        self.dist = dist
        self.t = int(t)
        self.grid = np.asarray(grid, dtype=float)
        self.A, self.b, self.row_scale = moment_constraints(dist, self.t, self.grid)
        self.n = len(self.grid)
        self.A_ub = None          # extra inequality rows on w (confidence set)
        self.b_ub = None

    def _ub(self, extra_A=None, extra_b=None, pad: int = 0):
        """Stack the set's own inequality rows with a caller's, padding for
        auxiliary variables (the mu block of the lower-bound LP)."""
        blocks, rhs = [], []
        if self.A_ub is not None:
            A = self.A_ub
            if pad:
                A = np.hstack([A, np.zeros((A.shape[0], pad))])
            blocks.append(A); rhs.append(self.b_ub)
        if extra_A is not None:
            blocks.append(np.atleast_2d(extra_A)); rhs.append(np.atleast_1d(extra_b))
        if not blocks:
            return None, None
        return np.vstack(blocks), np.concatenate(rhs)

    @classmethod
    def from_confidence_region(cls, region, t: int, grid: np.ndarray):
        """M_t^alpha: grid distributions whose moments lie in `region`.

        The only equality is sum(w) = 1; the moment conditions become the
        region's box inequalities, so the set has interior and cannot collapse
        to a point the way the projected set does.
        """
        from .confidence_set import box_constraints
        obj = cls.__new__(cls)
        obj.dist = None
        obj.t = int(t)
        obj.grid = np.asarray(grid, dtype=float)
        obj.n = len(obj.grid)
        obj.A = np.ones((1, obj.n))
        obj.b = np.array([1.0])
        obj.row_scale = np.array([1.0])
        obj.A_ub, obj.b_ub = box_constraints(region, obj.grid, int(t))
        obj.region = region
        return obj

    @classmethod
    def from_moments(cls, moments, t: int, grid: np.ndarray):
        """Build M_t from an *estimated* moment vector m_0..m_{2t-1}.

        This is what a timing rule actually has: the planner never sees the true
        distribution, only sample moments recovered from the observed cohort
        counts.  Estimated moments may fall outside the truncated moment cone, in
        which case M_t is empty and the LPs are infeasible -- that is a genuine
        no-estimate outcome for any rule built on the identified set, not an
        implementation error, and callers must record it as such.
        """
        obj = cls.__new__(cls)
        obj.dist = None
        obj.t = int(t)
        obj.grid = np.asarray(grid, dtype=float)
        m = np.asarray(moments, dtype=float)
        n_mom = 2 * t - 1
        A = np.vstack([obj.grid ** j for j in range(n_mom + 1)])
        b = np.array([1.0] + [m[j] for j in range(1, n_mom + 1)])
        row_scale = np.array([abs(x) if abs(x) > 0 else 1.0 for x in b])
        obj.A, obj.b, obj.row_scale = A / row_scale[:, None], b / row_scale, row_scale
        obj.n = len(obj.grid)
        obj.A_ub = None
        obj.b_ub = None
        return obj

    # -- feasibility ---------------------------------------------------------
    def feasible_point(self):
        """Any Q in M_t (phase-1 LP).  Returns (weights, status)."""
        Au, bu = self._ub()
        res = linprog(c=np.zeros(self.n), A_ub=Au, b_ub=bu,
                      A_eq=self.A, b_eq=self.b,
                      bounds=[(0.0, None)] * self.n, method=SOLVER)
        return (res.x if res.success else None), res.message

    def moment_residual(self, w) -> float:
        """Max absolute moment residual, reported in *unscaled* units."""
        return float(np.max(np.abs((self.A @ w - self.b) * self.row_scale)))

    def moment_residual_relative(self, w) -> float:
        """Max residual relative to each moment's own magnitude."""
        return float(np.max(np.abs(self.A @ w - self.b)))

    # -- upper bound ---------------------------------------------------------
    def _max_for_allocation(self, UTIL, PHI, kstar: int, f: float, beta: float):
        c = UTIL[kstar + 1:].sum(axis=0) + f * UTIL[kstar]
        bud = PHI[kstar + 1:].sum(axis=0) + f * PHI[kstar]
        Au, bu = self._ub(bud[None, :], [beta])
        res = linprog(c=-c, A_ub=Au, b_ub=bu,
                      A_eq=self.A, b_eq=self.b,
                      bounds=[(0.0, None)] * self.n, method=SOLVER)
        if not res.success:
            return -np.inf, None, res.message
        return float(c @ res.x), res.x, res.message

    def max_welfare(self, s: int, gamma: int, T: int, beta: float,
                    seed_profile=None, n_fill: int = 21, n_refine: int = 9,
                    n_rounds: int = 3):
        """Inner approximation of max_{Q in M_t} W^s(Q).

        The cutoff/fill pair is not a two-dimensional search.  Ordering the pool
        from the top y-group downwards, the treated set grows monotonically with
        the single scalar

            theta = (s - k*) + f  in  [0, s+1],

        so (k*, f) = (s - floor(theta), theta - floor(theta)) traces one nested
        family of allocations.  Scanning that scalar -- coarsely, then refining
        around the incumbent for `n_rounds` -- resolves the fill fraction far
        finer than a per-cutoff grid at a fraction of the LP count.
        """
        PHI, UTIL = group_matrices(self.grid, s, gamma, T)
        best = (-np.inf, None, -1, np.nan, "")
        n_lp = 0

        def evaluate(theta: float):
            nonlocal best, n_lp
            k, f = theta_to_cutoff(theta, s)
            val, w, msg = self._max_for_allocation(UTIL, PHI, k, f, beta)
            n_lp += 1
            if val > best[0]:
                best = (val, w, k, f, msg)

        if seed_profile is not None:                    # guarantees containment
            k0, f0 = seed_profile
            val, w, msg = self._max_for_allocation(UTIL, PHI, int(k0), float(f0), beta)
            n_lp += 1
            best = (val, w, int(k0), float(f0), msg)

        # Bracket the scan.  At theta in [j, j+1) the allocation treats the top j
        # y-groups in full, so it is feasible for some Q in M_t only if
        # min_Q sum_{k>s-j} m_k(Q) <= beta.  Once that fails at some j it fails
        # for every larger theta, so the scan can stop there -- rigorously, not
        # heuristically.  At these budget ratios only the top few groups are ever
        # affordable, which turns an 80-LP scan at T=20 into about a dozen.
        hi_theta = float(s + 1)
        for j in range(1, s + 2):
            mass = PHI[s - j + 1:].sum(axis=0)
            Au, bu = self._ub()
            res = linprog(c=mass, A_ub=Au, b_ub=bu, A_eq=self.A, b_eq=self.b,
                          bounds=[(0.0, None)] * self.n, method=SOLVER)
            n_lp += 1
            if not res.success:
                break                      # never narrow on a solver failure
            if res.fun > beta + 1e-15:
                hi_theta = float(j)
                break

        n_coarse = max(int(n_fill), int(4 * hi_theta) + 1)
        thetas = np.linspace(0.0, hi_theta, n_coarse)
        for th in thetas:
            evaluate(float(th))

        step = hi_theta / (n_coarse - 1)
        for _ in range(int(n_rounds)):
            centre = (s - best[2]) + best[3]
            lo, hi = max(0.0, centre - step), min(hi_theta, centre + step)
            for th in np.linspace(lo, hi, int(n_refine)):
                evaluate(float(th))
            step /= (int(n_refine) - 1) / 2.0
        return best, n_lp

    # -- lower bound ---------------------------------------------------------
    def _min_for_lambda(self, UTIL, PHI, lam: float, beta: float):
        """LP in (w, mu): min beta*lam + sum_k mu_k  s.t.  mu_k >= v_k - lam*m_k."""
        s1 = PHI.shape[0]
        nv = self.n + s1
        c = np.concatenate([np.zeros(self.n), np.ones(s1)])
        # -(UTIL - lam*PHI) w - mu <= 0
        A_ub, b_ub = self._ub(np.hstack([UTIL - lam * PHI, -np.eye(s1)]),
                              np.zeros(s1), pad=s1)
        A_eq = np.hstack([self.A, np.zeros((self.A.shape[0], s1))])
        res = linprog(c=c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=self.b,
                      bounds=[(0.0, None)] * nv, method=SOLVER)
        if not res.success:
            return np.inf, None, res.message
        return float(beta * lam + res.fun), res.x[:self.n], res.message

    def min_welfare(self, s: int, gamma: int, T: int, beta: float,
                    seed_lambda=None, n_lambda: int = 41, n_refine: int = 9,
                    n_rounds: int = 3):
        """Inner approximation of min_{Q in M_t} W^s(Q)."""
        PHI, UTIL = group_matrices(self.grid, s, gamma, T)
        best = (np.inf, None, np.nan, "")
        n_lp = 0

        lams = list(np.linspace(0.0, 1.0, n_lambda))
        if seed_lambda is not None and np.isfinite(seed_lambda):
            lams.insert(0, float(seed_lambda))          # guarantees containment

        for lam in lams:
            val, w, msg = self._min_for_lambda(UTIL, PHI, float(lam), beta)
            n_lp += 1
            if val < best[0]:
                best = (val, w, float(lam), msg)

        step = 1.0 / (n_lambda - 1)
        for _ in range(int(n_rounds)):
            lo, hi = max(0.0, best[2] - step), min(1.0, best[2] + step)
            for lam in np.linspace(lo, hi, int(n_refine)):
                val, w, msg = self._min_for_lambda(UTIL, PHI, float(lam), beta)
                n_lp += 1
                if val < best[0]:
                    best = (val, w, float(lam), msg)
            step /= (int(n_refine) - 1) / 2.0
        return best, n_lp

    # -- both ----------------------------------------------------------------
    def bounds(self, s: int, gamma: int, T: int, beta: float,
               truth_profile=None, truth_lambda=None,
               n_fill: int = 21, n_lambda: int = 41, n_refine: int = 9,
               n_rounds: int = 3) -> LPResult:
        (hi, w_hi, kstar, fstar, msg_hi), n1 = self.max_welfare(
            s, gamma, T, beta, seed_profile=truth_profile,
            n_fill=n_fill, n_refine=n_refine, n_rounds=n_rounds)
        (lo, w_lo, lam, msg_lo), n2 = self.min_welfare(
            s, gamma, T, beta, seed_lambda=truth_lambda,
            n_lambda=n_lambda, n_refine=n_refine, n_rounds=n_rounds)
        return LPResult(
            lower=lo, upper=hi,
            lower_status="optimal" if np.isfinite(lo) else msg_lo,
            upper_status="optimal" if np.isfinite(hi) else msg_hi,
            lower_residual=self.moment_residual(w_lo) if w_lo is not None else np.nan,
            upper_residual=self.moment_residual(w_hi) if w_hi is not None else np.nan,
            n_lp_solved=n1 + n2,
            lambda_star=lam, cutoff_star=kstar, fill_star=fstar,
            min_weight=float(np.min(w_hi)) if w_hi is not None else np.nan,
        )


def truth_seeds(dist, s: int, gamma: int, T: int, beta: float):
    """The truth's own (cutoff, fill) and dual price lambda.

    Seeding both scans with these makes truth containment structural.
    """
    from .welfare import group_masses, group_utilities
    m = group_masses(dist, s, gamma)
    v = group_utilities(dist, s, gamma, T)
    _, a = fill_greedy(m, v, beta)
    k, f = cutoff_and_fill(a)
    lam = float(v[k] / m[k]) if m[k] > 0 else np.nan
    return (k, f), lam
