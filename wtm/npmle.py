"""Nonparametric MLE of the risk distribution over a support grid (Lindsay 1983).

The likelihood is the exact multinomial over observable single-individual
records.  Writing Psi[a, i] = psi_a(g_i) for record atom `a` and grid point `i`,
and n_a for the observed counts,

    log L(w) = sum_a n_a log( (Psi w)_a ),     w >= 0,  sum_i w_i = 1.

Since sum_a psi_a(p) = 1 for every p, this is a standard finite mixture
likelihood and the EM update is the classical one:

    w_i  <-  w_i / N * sum_a n_a Psi[a,i] / (Psi w)_a

which is monotone in the likelihood and keeps w on the simplex automatically.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from .observation import individual_functionals


@dataclass
class NPMLEResult:
    weights: np.ndarray
    loglik: float
    iterations: int
    converged: bool
    tol: float
    grid_size: int
    n_support: int


def record_matrix(grid: np.ndarray, t: int, gamma: int = 1) -> np.ndarray:
    """Psi[a, i] = P(record atom a | p = grid_i).  Columns sum to 1."""
    return np.vstack([f.evaluate(grid) for _, f in individual_functionals(t, gamma)])


def fit_npmle(counts: np.ndarray, grid: np.ndarray, t: int, gamma: int = 1,
              tol: float = 1e-8, max_iter: int = 5000,
              Psi: np.ndarray = None, w0: np.ndarray = None) -> NPMLEResult:
    """EM for the mixing distribution.  `counts` are record-atom counts.

    `w0` warm-starts the iteration.  The log-likelihood is concave in w, so the
    optimum is unique and the starting point cannot change where EM converges --
    only how long it takes.  Warm-starting each bootstrap replicate from the
    primary fit is therefore free accuracy-wise and large computationally,
    because bootstrap resamples sit close to the original data.
    """
    Psi = record_matrix(grid, t, gamma) if Psi is None else Psi
    n = np.asarray(counts, dtype=float)
    N = n.sum()
    if w0 is None:
        w = np.full(len(grid), 1.0 / len(grid))
    else:
        w = np.array(w0, dtype=float)
        w = np.maximum(w, 1e-300)
        w = w / w.sum()

    prev = -np.inf
    it = 0
    converged = False
    for it in range(1, int(max_iter) + 1):
        mix = Psi @ w                                  # (n_atoms,)
        mix = np.maximum(mix, 1e-300)
        ll = float(np.sum(n * np.log(mix)))
        w = w * (Psi.T @ (n / mix)) / N
        s = w.sum()
        if s > 0:
            w = w / s
        # The check must not fire on the first pass: with prev = -inf the test
        # `ll - prev <= tol * abs(prev)` reads inf <= inf and exits immediately.
        if np.isfinite(prev) and (ll - prev) <= tol * max(1.0, abs(prev)):
            converged = True
            break
        prev = ll
    mix = np.maximum(Psi @ w, 1e-300)
    return NPMLEResult(weights=w, loglik=float(np.sum(n * np.log(mix))),
                       iterations=it, converged=converged, tol=tol,
                       grid_size=len(grid), n_support=int((w > 1e-10).sum()))


def fit_npmle_batch(counts_matrix: np.ndarray, grid: np.ndarray, t: int,
                    gamma: int = 1, tol: float = 1e-8, max_iter: int = 20000,
                    Psi: np.ndarray = None, w0: np.ndarray = None):
    """EM for many count vectors at once.

    Identical update to `fit_npmle`, applied to all replicates simultaneously so
    each iteration is one (E x G) @ (G x A) product instead of E tiny ones.  A
    replicate stops being updated once its own log-likelihood increment falls
    below `tol`, so the answer it lands on is the same one the sequential loop
    would reach; only the scheduling differs.

    Returns (weights (E, G), converged (E,), iterations run).
    """
    Psi = record_matrix(grid, t, gamma) if Psi is None else Psi
    n = np.asarray(counts_matrix, dtype=float)
    E, G = n.shape[0], len(grid)
    N = n.sum(axis=1, keepdims=True)
    W = (np.tile(w0, (E, 1)) if w0 is not None
         else np.full((E, G), 1.0 / G))
    W = np.maximum(W, 1e-300)
    W /= W.sum(axis=1, keepdims=True)

    prev = np.full(E, -np.inf)
    done = np.zeros(E, dtype=bool)
    it = 0
    for it in range(1, int(max_iter) + 1):
        active = ~done
        if not active.any():
            break
        mix = np.maximum(W[active] @ Psi.T, 1e-300)
        ll = (n[active] * np.log(mix)).sum(axis=1)
        upd = W[active] * ((n[active] / mix) @ Psi) / N[active]
        upd = np.maximum(upd, 0.0)
        upd /= np.maximum(upd.sum(axis=1, keepdims=True), 1e-300)
        W[active] = upd
        p = prev[active]
        newly = np.isfinite(p) & ((ll - p) <= tol * np.maximum(1.0, np.abs(p)))
        idx = np.where(active)[0]
        done[idx[newly]] = True
        prev[idx] = ll
    return W, done, it


def fit_npmle_fast(counts: np.ndarray, grid: np.ndarray, t: int, gamma: int = 1,
                   tol: float = 1e-8, max_iter: int = 20000,
                   Psi: np.ndarray = None, w0: np.ndarray = None,
                   polish: bool = True) -> NPMLEResult:
    """SQUAREM-accelerated EM (Varadhan & Roland 2008), same fixed point.

    Why acceleration rather than a different optimizer.  `Psi` is (n_atoms x G)
    with n_atoms << G and, empirically, rank 5 at t = 2 -- so the log-likelihood
    is concave but NOT strictly concave in `w`, and its set of maximizers is a
    polytope rather than a point.  The decision depends on the full `w` through
    the welfare trajectory, so a different solver could return a different member
    of that set and change the selected date.  SQUAREM accelerates the EM map
    itself, so it converges to the same fixed point EM would reach; only the
    number of iterations changes.

    Each cycle costs three EM maps and extrapolates along the secant, giving a
    superlinear step where plain EM crawls.  A monotonicity safeguard backtracks
    the step length whenever the extrapolation would lower the log-likelihood, so
    the iteration is never worse than EM.  With `polish=True` the routine finishes
    with plain EM steps until EM's own stopping rule fires, so the returned point
    satisfies exactly the criterion `fit_npmle` uses.
    """
    Psi = record_matrix(grid, t, gamma) if Psi is None else Psi
    n = np.asarray(counts, dtype=float)
    N = n.sum()
    if w0 is None:
        w = np.full(len(grid), 1.0 / len(grid))
    else:
        w = np.maximum(np.array(w0, dtype=float), 1e-300)
        w = w / w.sum()

    def em(x):
        mix = np.maximum(Psi @ x, 1e-300)
        y = x * (Psi.T @ (n / mix)) / N
        s = y.sum()
        return y / s if s > 0 else y

    def ll_of(x):
        return float(np.sum(n * np.log(np.maximum(Psi @ x, 1e-300))))

    def project(x):
        x = np.maximum(x, 0.0)
        s = x.sum()
        return x / s if s > 0 else np.full(len(x), 1.0 / len(x))

    evals = 0
    prev = -np.inf
    converged = False
    while evals < int(max_iter):
        w1 = em(w)
        w2 = em(w1)
        evals += 2
        r = w1 - w
        v = (w2 - w1) - r
        vn = np.linalg.norm(v)
        if vn < 1e-300:
            w = w2
        else:
            alpha = -np.linalg.norm(r) / vn
            ll2 = ll_of(w2)
            cand = project(w - 2.0 * alpha * r + (alpha ** 2) * v)
            cand = em(cand)
            evals += 1
            # monotonicity safeguard: never accept a step that loses likelihood
            back = 0
            while ll_of(cand) < ll2 and back < 8:
                alpha = (alpha - 1.0) / 2.0
                cand = project(w - 2.0 * alpha * r + (alpha ** 2) * v)
                cand = em(cand)
                evals += 1
                back += 1
            w = cand if ll_of(cand) >= ll2 else w2
        ll = ll_of(w)
        if np.isfinite(prev) and (ll - prev) <= tol * max(1.0, abs(prev)):
            converged = True
            break
        prev = ll

    if polish and converged:
        # finish under EM's own stopping rule so the returned point satisfies
        # exactly the criterion the frozen code uses
        prev = -np.inf
        for _ in range(2000):
            mix = np.maximum(Psi @ w, 1e-300)
            ll = float(np.sum(n * np.log(mix)))
            w = em(w)
            evals += 1
            if np.isfinite(prev) and (ll - prev) <= tol * max(1.0, abs(prev)):
                break
            prev = ll

    mix = np.maximum(Psi @ w, 1e-300)
    return NPMLEResult(weights=w, loglik=float(np.sum(n * np.log(mix))),
                       iterations=evals, converged=converged, tol=tol,
                       grid_size=len(grid), n_support=int((w > 1e-10).sum()))


def fit_npmle_certified(counts: np.ndarray, grid: np.ndarray, t: int,
                        gamma: int = 1, delta: float = 1e-10,
                        Psi: np.ndarray = None, max_outer: int = 400,
                        inner_iter: int = 200000, active0: np.ndarray = None):
    """NPMLE with a KKT convergence CERTIFICATE (support-reduction / VDM).

    The frozen `fit_npmle` stops when the log-likelihood increment is small,
    which on a sublinearly-converging EM map fires far from the optimum: its
    Lindsay gradient is measurably positive.  This routine instead stops on the
    optimality condition itself.

    Lindsay's directional derivative

        D(p) = sum_a n_a Psi[a, p] / (Psi w)_a  -  N

    satisfies D(p) <= 0 for all p, with equality on the support, if and only if
    `w` maximises the likelihood.  So `sup_p D(p) / N` is a certificate of
    suboptimality that can be checked, not assumed.

    Algorithm: repeatedly add the most-violating grid point to an active set and
    re-solve the (tiny) restricted problem to high precision.  Because `Psi` has
    few rows, the optimal support is small, so the restricted problems stay
    cheap and the outer loop terminates in a handful of steps.

    Returns `(NPMLEResult, certificate)` where `certificate` is `sup_p D(p) / N`.
    """
    Psi = record_matrix(grid, t, gamma) if Psi is None else Psi
    n = np.asarray(counts, dtype=float)
    N = n.sum()
    G = Psi.shape[1]

    def grad(w):
        mix = np.maximum(Psi @ w, 1e-300)
        return Psi.T @ (n / mix) - N

    if active0 is not None and len(active0):
        # Warm start: seed the active set from a nearby solution (e.g. the point
        # fit, when solving bootstrap replicates).  This changes only where the
        # support-reduction search BEGINS.  Termination is still governed by the
        # KKT certificate `sup_p D(p)/N <= delta` and by the support invariant,
        # so a warm-started solve is accepted on exactly the same terms as a
        # cold one -- it cannot be certified unless it satisfies the same
        # optimality condition.
        active = np.unique(np.asarray(active0, dtype=int))
    else:
        w = np.full(G, 1.0 / G)
        w = w * (Psi.T @ (n / np.maximum(Psi @ w, 1e-300))) / N   # one EM step
        active = np.flatnonzero(w > w.max() * 1e-6)[:8]
        if len(active) == 0:
            active = np.array([int(np.argmax(Psi.sum(axis=0)))])

    cert = np.inf
    prev_v = np.array([])
    for outer in range(1, int(max_outer) + 1):
        # --- solve the restricted problem on `active` exactly.  The active set
        # is tiny (the optimal support is bounded by the number of record
        # atoms), so a constrained Newton-type solve is both fast and accurate,
        # where EM on the same subproblem still crawls.
        P = Psi[:, active]
        k = len(active)
        v0 = np.full(k, 1.0 / k)
        if outer > 1 and len(prev_v) == k - 1:
            v0 = np.append(prev_v * (1.0 - 1.0 / k), 1.0 / k)

        # The objective is normalised by N.  Without this the inner solver's
        # ftol is applied to a quantity of order N*log(...), so at N = 1e5 the
        # achievable absolute precision sits right at the certificate threshold,
        # the residual never falls below delta, and the outer loop runs to its
        # cap -- 1300x SLOWER than EM instead of faster.  Normalising makes the
        # inner tolerance scale-free, so cost no longer grows with the data.
        def _neg_ll(v):
            return -float(np.sum(n * np.log(np.maximum(P @ v, 1e-300)))) / N

        def _neg_ll_grad(v):
            return -(P.T @ (n / np.maximum(P @ v, 1e-300))) / N

        sol = minimize(_neg_ll, v0, jac=_neg_ll_grad, method="SLSQP",
                       bounds=[(0.0, 1.0)] * k,
                       constraints=[{"type": "eq",
                                     "fun": lambda z: z.sum() - 1.0,
                                     "jac": lambda z: np.ones_like(z)}],
                       options={"maxiter": int(inner_iter), "ftol": 1e-14})
        v = np.maximum(np.asarray(sol.x, dtype=float), 0.0)
        sv = v.sum()
        v = v / sv if sv > 0 else np.full(k, 1.0 / k)

        prev_v = v
        w = np.zeros(G)
        w[active] = v
        d = grad(w)
        cert = float(d.max() / N)
        if cert <= delta:
            break
        j = int(np.argmax(d))
        if j in set(active.tolist()):
            break                       # cannot improve; numerically stalled
        active = np.append(active[v > 1e-14], j)

    mix = np.maximum(Psi @ w, 1e-300)
    n_sup = int((w > 1e-10).sum())
    # Lindsay: the NPMLE support is bounded by the number of observable atoms.
    # A violation means the returned point is not a solution -- exactly the
    # failure mode the frozen EM exhibits (42 atoms against a bound of 8).
    invariant_ok = n_sup <= Psi.shape[0]
    res = NPMLEResult(weights=w, loglik=float(np.sum(n * np.log(mix))),
                      iterations=outer,
                      converged=bool(cert <= delta and invariant_ok),
                      tol=delta, grid_size=G, n_support=n_sup)
    return res, cert
