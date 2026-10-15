"""Batch EM trajectories with per-iteration diagnostics.

The EM update is the one in `wtm.npmle.fit_npmle`, applied to many count
vectors at once and recorded on a schedule:

    mix  = max(Psi w, 1e-300)
    w   <- w * (Psi^T (n / mix)) / N,      then  w <- w / sum(w)

`w_k` denotes the state after exactly k updates; `w_0` is the initialization.
`fit_npmle(..., tol=1e-300, max_iter=K)` returns `w_K` (it performs K updates
unless its own stopping test fires, which with tol=1e-300 requires an exactly
zero log-likelihood increment).  validate.py checks this equivalence on the
archived benchmark datasets.

Everything is evaluated at the state `w_k` *before* the k-th update is
applied, from the same `mix` the update uses:

    ll_k   = sum_a (n_a/N) log mix_a                 (per-individual log likelihood)
    kkt_k  = max_g [ sum_a n_a Psi[a,g] / mix_a - N ] / N
           = the certificate `sup_p D(p)/N` of `fit_npmle_certified`.

By concavity, ll* - ll_k <= kkt_k, so the KKT residual is a certified upper
bound on the likelihood gap to the grid optimum.  This is what lets the study
report convergence by the existing optimality condition instead of by a small
successive likelihood change.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import stats

from wtm.lp import group_matrices
from wtm.npmle import fit_npmle_certified, record_matrix
from wtm.observation import individual_functionals
from wtm.regret_calibrated import _trajectory

DELTA = 1e-8            # existing certification tolerance (fit_npmle_certified)
FLOOR_EPS = 0.01        # non-uniform initializations: (1-eps)*shape + eps*uniform
MONO_TOL = 1e-13        # per-individual log-likelihood units
TIE_REL = 1e-10         # near-tie threshold that triggers the exact fallback
# Weights are floored at 1e-300 after every update.  Without it, grid points far
# in the tail decay into subnormal doubles, which makes each iteration up to
# ~10x slower.  1e-300 is a normal double, the same floor fit_npmle applies to
# `mix`, and it changes no recorded quantity at double precision; it also keeps
# every coordinate strictly positive (revivable), as EM requires.
WEIGHT_FLOOR = 1e-300

# Initialization family.  Beta shapes are integrated over grid bins (Voronoi
# cells of the grid points).  Random initializations are Dirichlet(1) draws on
# the grid.  All non-uniform shapes are mixed with the uniform at FLOOR_EPS, so
# every coordinate is >= FLOOR_EPS / G > 0 (EM cannot revive a zero weight).
BETA_INITS = {"low": (1.0, 19.0), "middle": (4.0, 4.0), "high": (5.0, 2.0)}
RANDOM_INITS = {"rand1": 20261001, "rand2": 20261002, "rand3": 20261003, "rand4": 20261004}
INIT_NAMES = ("uniform", "low", "middle", "high") + tuple(RANDOM_INITS)


def bin_edges(grid: np.ndarray) -> np.ndarray:
    """Edges of the grid's Voronoi cells, clipped to [0, 1]."""
    g = np.asarray(grid, dtype=float)
    return np.concatenate([[0.0], 0.5 * (g[1:] + g[:-1]), [1.0]])


def init_weights(name: str, grid: np.ndarray) -> np.ndarray:
    G = len(grid)
    if name == "uniform":
        return np.full(G, 1.0 / G)
    if name in BETA_INITS:
        a, b = BETA_INITS[name]
        mass = np.diff(stats.beta.cdf(bin_edges(grid), a, b))
    elif name in RANDOM_INITS:
        mass = np.random.default_rng(RANDOM_INITS[name]).dirichlet(np.ones(G))
    else:
        raise KeyError(name)
    mass = np.maximum(mass, 0.0)
    mass = mass / mass.sum()
    w = (1.0 - FLOOR_EPS) * mass + FLOOR_EPS / G
    return w / w.sum()


def bin_projection(dist, grid: np.ndarray) -> np.ndarray:
    """Weights placing P's mass on each grid cell at its grid point (realizable truth)."""
    m = np.diff(np.asarray(dist.cdf(bin_edges(grid)), dtype=float))
    m = np.maximum(m, 0.0)
    return m / m.sum()


def schedule(cap: int, dense: int = 100, per_decade: int = 20) -> np.ndarray:
    """0..dense every iteration, then ~per_decade log-spaced points through cap."""
    s = set(range(0, min(dense, cap) + 1))
    if cap > dense:
        n = int(round(per_decade * np.log10(cap / dense))) + 1
        s |= set(np.unique(np.round(np.logspace(np.log10(dense), np.log10(cap), n))).astype(int).tolist())
    s.add(int(cap))
    return np.array(sorted(s), dtype=np.int64)


def exact_cell_probs(dist, tau: int) -> np.ndarray:
    """q_c = E_P[psi_c(p)] for the length-tau pre-failure window, from exact moments."""
    return np.array([f.expectation(dist) for _, f in individual_functionals(tau, 1)])


class Welfare:
    """Estimated welfare W^s, s = 1..T-1, for many weight vectors at once.

    Mirrors `wtm.welfare.fill_greedy` (top-down greedy fill, same branch
    structure).  Rows whose best and second-best dates are within TIE_REL are
    re-decided with the existing scalar `_trajectory`, so the selected date
    always follows the existing earliest-argmax rule on the existing code path.
    """

    def __init__(self, grid, T: int, beta: float, gamma: int = 1):
        self.T, self.beta = int(T), float(beta)
        self.cache = [group_matrices(grid, int(s), gamma, T) for s in range(1, T)]
        self._PT = [np.ascontiguousarray(P.T) for P, _ in self.cache]
        self._UT = [np.ascontiguousarray(U.T) for _, U in self.cache]
        self.n_fallback = 0

    def curves(self, W: np.ndarray) -> np.ndarray:
        B = W.shape[0]
        out = np.empty((B, self.T - 1))
        for i, (PT, UT) in enumerate(zip(self._PT, self._UT)):
            M, V = W @ PT, W @ UT
            rem = np.full(B, self.beta)
            A = np.zeros_like(M)
            for k in range(M.shape[1] - 1, -1, -1):
                m = M[:, k]
                act = (rem > 0.0) & (m > 0.0)
                take = np.where(act, np.minimum(rem, m), 0.0)
                A[:, k] = np.where(act, take / np.where(m > 0.0, m, 1.0), 0.0)
                rem = rem - take
            out[:, i] = (A * V).sum(axis=1)
        return out

    def decide(self, W: np.ndarray, C: np.ndarray) -> np.ndarray:
        d = np.argmax(C, axis=1) + 1
        if C.shape[1] > 1:
            srt = np.sort(C, axis=1)
            top, sec = srt[:, -1], srt[:, -2]
            near = (top - sec) <= TIE_REL * np.maximum(np.abs(top), 1e-300)
            for r in np.flatnonzero(near):
                self.n_fallback += 1
                d[r] = int(np.argmax(_trajectory(W[r], self.cache, self.beta))) + 1
        return d


@dataclass
class EMRun:
    iters: np.ndarray                      # (R,)
    ll: np.ndarray                         # (R, B) per-individual log likelihood
    kkt: np.ndarray                        # (R, B)
    qhat: np.ndarray                       # (R, B, A) fitted cell probabilities
    mom: np.ndarray                        # (R, B, J) raw moments 1..J
    curves: dict                           # key -> (R, B, T-1)
    dates: dict                            # key -> (R, B)
    min_w: np.ndarray                      # (R, B)
    sum_dev: np.ndarray                    # (R, B) |sum w - 1|
    first_cert: np.ndarray                 # (B,) first k with kkt <= DELTA, -1 if never
    mono_viol: np.ndarray                  # (B,) count of k with ll_k < ll_{k-1} - MONO_TOL
    max_drop: np.ndarray                   # (B,) largest single-step decrease
    finite: np.ndarray                     # (B,)
    snapshots: dict = field(default_factory=dict)   # k -> (B, G) weights
    seconds: float = 0.0


def run_em(counts: np.ndarray, Psi: np.ndarray, grid: np.ndarray, W0: np.ndarray,
           sched: np.ndarray, evaluators: dict, n_moments: int = 8,
           snapshot_at=()) -> EMRun:
    """Run EM for every row of `counts` from the matching row of `W0`."""
    import time
    t0 = time.perf_counter()
    n = np.ascontiguousarray(np.atleast_2d(np.asarray(counts, dtype=float)))
    B, A = n.shape
    N = n.sum(axis=1, keepdims=True)
    W = np.ascontiguousarray(np.array(W0, dtype=float))
    assert W.shape == (B, len(grid))
    PsiT = np.ascontiguousarray(Psi.T)
    pows = np.vstack([grid ** j for j in range(1, n_moments + 1)]).T   # (G, J)
    sched = np.asarray(sched, dtype=np.int64)
    cap = int(sched[-1])
    is_rec = np.zeros(cap + 1, dtype=bool)
    is_rec[sched] = True
    snaps = set(int(s) for s in snapshot_at)

    R = len(sched)
    ll_r = np.empty((R, B)); kkt_r = np.empty((R, B))
    q_r = np.empty((R, B, A)); mom_r = np.empty((R, B, n_moments))
    minw_r = np.empty((R, B)); sdev_r = np.empty((R, B))
    curves = {k: np.empty((R, B, ev.T - 1)) for k, ev in evaluators.items()}
    dates = {k: np.empty((R, B), dtype=np.int16) for k in evaluators}
    snapshots = {}

    first_cert = np.full(B, -1, dtype=np.int64)
    mono_viol = np.zeros(B, dtype=np.int64)
    max_drop = np.zeros(B)
    finite = np.ones(B, dtype=bool)
    ll_prev = np.full(B, -np.inf)
    mix = np.empty((B, A)); U = np.empty_like(W)
    r = 0
    for k in range(cap + 1):
        np.matmul(W, PsiT, out=mix)
        np.maximum(mix, 1e-300, out=mix)
        ratio = n / mix
        np.matmul(ratio, Psi, out=U)
        ll = (n * np.log(mix)).sum(axis=1) / N[:, 0]
        kkt = (U.max(axis=1) - N[:, 0]) / N[:, 0]
        drop = ll_prev - ll
        bad = drop > MONO_TOL
        mono_viol += bad
        np.maximum(max_drop, np.where(np.isfinite(drop), drop, 0.0), out=max_drop)
        newc = (first_cert < 0) & (kkt <= DELTA)
        first_cert[newc] = k
        ll_prev = ll
        if is_rec[k]:
            fin = np.isfinite(W).all(axis=1) & np.isfinite(ll)
            finite &= fin
            ll_r[r], kkt_r[r] = ll, kkt
            q_r[r] = W @ PsiT
            mom_r[r] = W @ pows
            minw_r[r] = W.min(axis=1)
            sdev_r[r] = np.abs(W.sum(axis=1) - 1.0)
            for key, ev in evaluators.items():
                C = ev.curves(W)
                curves[key][r] = C
                dates[key][r] = ev.decide(W, C)
            if k in snaps:
                snapshots[k] = W.copy()
            r += 1
        if k == cap:
            break
        W *= U
        W /= N
        W /= W.sum(axis=1, keepdims=True)
        np.maximum(W, WEIGHT_FLOOR, out=W)
    return EMRun(iters=sched, ll=ll_r, kkt=kkt_r, qhat=q_r, mom=mom_r, curves=curves,
                 dates=dates, min_w=minw_r, sum_dev=sdev_r, first_cert=first_cert,
                 mono_viol=mono_viol, max_drop=max_drop, finite=finite,
                 snapshots=snapshots, seconds=time.perf_counter() - t0)


def em_status(run: EMRun) -> np.ndarray:
    """Per-row terminal status.

    certified       final KKT residual <= DELTA (existing certificate)
    stagnated       not certified, and the log likelihood gained <= 10 ulp over
                    the last recorded decade of iterations (no measurable progress)
    cap_reached     not certified, still improving at the iteration cap
    numerical_failure  non-finite weights/likelihood, or invalid weights
    """
    k = run.iters
    last = len(k) - 1
    j = int(np.searchsorted(k, max(1, k[-1] // 10)))
    gain = run.ll[last] - run.ll[j]
    tiny = 10 * np.finfo(float).eps * np.maximum(1.0, np.abs(run.ll[last]))
    st = np.where(run.kkt[last] <= DELTA, "certified",
                  np.where(gain <= tiny, "stagnated", "cap_reached")).astype(object)
    invalid = (~run.finite) | (run.min_w.min(axis=0) < 0) | (run.sum_dev.max(axis=0) > 1e-9)
    st[invalid] = "numerical_failure"
    return st


@dataclass
class NPMLEFit:
    weights: np.ndarray
    ll: float              # per individual
    kkt: float
    n_support: int
    status: str            # certified | valid_uncertified | invalid
    seconds: float


def certified_npmle(counts_row, grid, tau, Psi, delta=DELTA) -> NPMLEFit:
    """`fit_npmle_certified` plus the existing validity rule.

    The archived benchmark (rerun_certified.py, matched_info_pairs.py) accepts a
    fit when its support respects Lindsay's bound and its likelihood is finite;
    a residual above delta is flagged but not rejected.  Both facts are kept.
    """
    import time
    t0 = time.perf_counter()
    n = np.asarray(counts_row, dtype=float)
    fit, cert = fit_npmle_certified(n, grid, tau, 1, delta=delta, Psi=Psi)
    mix = Psi @ fit.weights
    valid = bool(fit.n_support <= Psi.shape[0] and np.isfinite(fit.loglik)
                 and np.isfinite(mix).all())
    status = "invalid" if not valid else ("certified" if cert <= delta else "valid_uncertified")
    return NPMLEFit(weights=fit.weights, ll=float(fit.loglik / n.sum()), kkt=float(cert),
                    n_support=int(fit.n_support), status=status,
                    seconds=time.perf_counter() - t0)


def grid_and_psi(G: int, tau: int):
    grid = np.linspace(0.0, 1.0, int(G))
    return grid, record_matrix(grid, tau, 1)
