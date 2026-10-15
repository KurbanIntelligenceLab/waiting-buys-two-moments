"""The three plug-in timing rules (draft Section 6).

All three share one decision template -- estimate the risk distribution from the
observed exit records, reconstruct the welfare trajectory, act at its argmax --
and differ only in the estimator.  That is why the draft groups them: the
comparison isolates the estimator, holding the decision principle fixed.

  parametric_beta   match the first two moments in closed form (Shirali et al.)
  nnls              nonnegative least squares on the record law (ablation)
  npmle             nonparametric MLE over the grid (Lindsay 1983)

The parametric rule is the one that can fail outright: the method-of-moments
system has no Beta solution when the estimated variance is not strictly inside
(0, mu(1-mu)), and on realised data it often is not.  That is a genuine
no-estimate, recorded and scored, never silently dropped.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import nnls

from .distributions import BetaRisk
from .lp import group_matrices
from .npmle import fit_npmle, record_matrix
from .welfare import fill_greedy

GRID_SIZE = 200


@dataclass
class PluginOutput:
    selected_date: int
    success: bool = True
    failure_reason: str = ""
    estimator: str = ""
    trajectory: np.ndarray = field(repr=False, default=None)
    fitted: object = field(repr=False, default=None)
    n_iterations: int = 0
    converged: bool = True


def _trajectory_from_weights(w, grid, T, gamma, beta):
    return np.array([fill_greedy(P @ w, U @ w, beta)[0]
                     for P, U in (group_matrices(grid, int(s), gamma, T)
                                  for s in range(1, T))])


def _argmax_date(W):
    return int(np.argmax(W)) + 1


# -- 1. parametric Beta, method of moments -----------------------------------
def parametric_beta(record_counts, T, gamma, t, beta, moments=None,
                    grid_size=GRID_SIZE) -> PluginOutput:
    """Beta(alpha, beta) from the first two moments, in closed form."""
    if moments is None:
        return PluginOutput(-1, False, "moments_not_supplied", "parametric_beta")
    m1, m2 = float(moments[1]), float(moments[2])
    v = m2 - m1 * m1
    if not (0.0 < m1 < 1.0):
        return PluginOutput(-1, False, "mean_outside_unit_interval",
                            "parametric_beta")
    if v <= 0:
        return PluginOutput(-1, False, "nonpositive_variance", "parametric_beta")
    if v >= m1 * (1.0 - m1):
        return PluginOutput(-1, False, "variance_exceeds_bernoulli_bound",
                            "parametric_beta")
    c = m1 * (1.0 - m1) / v - 1.0
    a, b = m1 * c, (1.0 - m1) * c
    if not (np.isfinite(a) and np.isfinite(b) and a > 0 and b > 0):
        return PluginOutput(-1, False, "invalid_beta_parameters",
                            "parametric_beta")
    dist = BetaRisk(a, b)
    grid = np.linspace(0.0, 1.0, grid_size)
    dens = np.exp((a - 1) * np.log(np.clip(grid, 1e-12, 1))
                  + (b - 1) * np.log(np.clip(1 - grid, 1e-12, 1)))
    if not np.isfinite(dens).all() or dens.sum() <= 0:
        return PluginOutput(-1, False, "degenerate_beta_density",
                            "parametric_beta")
    w = dens / dens.sum()
    W = _trajectory_from_weights(w, grid, T, gamma, beta)
    if not np.isfinite(W).all() or W.max() <= 0:
        return PluginOutput(-1, False, "degenerate_trajectory", "parametric_beta")
    return PluginOutput(_argmax_date(W), True, "", "parametric_beta",
                        trajectory=W, fitted=dist)


# -- 2. nonnegative least squares --------------------------------------------
def nnls_plugin(record_counts, T, gamma, t, beta,
                grid_size=GRID_SIZE) -> PluginOutput:
    """Grid weights by NNLS against the observed record frequencies."""
    grid = np.linspace(0.0, 1.0, grid_size)
    Psi = record_matrix(grid, t, gamma)
    n = np.asarray(record_counts, dtype=float)
    q = n / n.sum()
    big = 10.0 * max(1.0, np.abs(Psi).max())
    A = np.vstack([Psi, big * np.ones(len(grid))])
    b = np.concatenate([q, [big]])
    w, _ = nnls(A, b)
    s = w.sum()
    if s <= 0 or not np.isfinite(s):
        return PluginOutput(-1, False, "nnls_returned_empty", "nnls")
    w = w / s
    W = _trajectory_from_weights(w, grid, T, gamma, beta)
    if not np.isfinite(W).all() or W.max() <= 0:
        return PluginOutput(-1, False, "degenerate_trajectory", "nnls")
    return PluginOutput(_argmax_date(W), True, "", "nnls", trajectory=W, fitted=w)


# -- 3. nonparametric MLE ----------------------------------------------------
def npmle_plugin(record_counts, T, gamma, t, beta, grid_size=GRID_SIZE,
                 tol=1e-8, max_iter=20000) -> PluginOutput:
    """NPMLE over the grid, using the exact record multinomial."""
    grid = np.linspace(0.0, 1.0, grid_size)
    Psi = record_matrix(grid, t, gamma)
    fit = fit_npmle(np.asarray(record_counts, dtype=float), grid, t, gamma,
                    tol=tol, max_iter=max_iter, Psi=Psi)
    if not fit.converged:
        return PluginOutput(-1, False, "npmle_no_convergence", "npmle",
                            n_iterations=fit.iterations, converged=False)
    W = _trajectory_from_weights(fit.weights, grid, T, gamma, beta)
    if not np.isfinite(W).all() or W.max() <= 0:
        return PluginOutput(-1, False, "degenerate_trajectory", "npmle",
                            n_iterations=fit.iterations)
    return PluginOutput(_argmax_date(W), True, "", "npmle", trajectory=W,
                        fitted=fit.weights, n_iterations=fit.iterations)
