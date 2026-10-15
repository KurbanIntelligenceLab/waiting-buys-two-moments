"""Covariance-weighted projection of an estimated moment vector onto the
truncated Hausdorff moment cone of [0,1]  (resolves NEEDS_DECISION D8c).

Why this exists.  The least-squares moment estimate recovered from cohort counts
is unbiased but noisy, and its high-order entries are far below the noise floor:
at t = 4 the true `m_7 ~ 1.6e-4` while the sampling error in the cohort counts is
`~1/sqrt(N)`.  The estimate routinely contains **negative** high moments, which no
distribution on [0,1] can produce, so `M_t` is empty and every identified-set
rule returns nothing.  Projecting first repairs feasibility without borrowing the
NPMLE's parametric-ish structure, which would couple the worst-case controls to
the plug-in family they are meant to control for.

The projection.  Any distribution supported on the grid has moments in the cone,
so it is enough to search over grid weights:

    m* = V w*,     w* = argmin_{w >= 0, sum w = 1} (V w - mhat)' Omega (V w - mhat)

with `V[j,i] = g_i^j`.  This is a convex QP; with `Omega = L'L` it is solved as a
nonnegative least-squares problem on the stacked system `[L V ; rho 1']`.

Weighting.  `Omega` is the (pseudo-)inverse of the sampling covariance of `mhat`,
so a moment estimated to within 1e-6 is not traded off against one estimated to
within 1e-2.  With the identity weight the projection would be dominated by the
low-order moments purely because they are larger, and would drag the well
estimated ones around to rescue the badly estimated ones.

Covariance.  The i.i.d. unit is an individual's full observation path, so the
path counts are exactly multinomial and

    mhat = B phat,        B = pinv(M) P',
    Cov(mhat) = (1/N) B ( diag(phat) - phat phat' ) B'

where `P` maps paths to cohort cells and `M` is the cohort design.  No asymptotic
approximation beyond the multinomial itself.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import nnls

from .moment_cone import build_path_alphabet, cohort_design, cone_statistic


# -- observing paths ---------------------------------------------------------
def path_offsets(t: int):
    """Start index of each block in `build_path_alphabet` ordering."""
    off, cur = {}, 0
    for e in range(1, t + 1):
        off[e] = cur
        cur += 2 ** e
    off[None] = cur
    return off


def path_indices(y_by_period: np.ndarray, exited_by: np.ndarray,
                 active_mask: np.ndarray, t: int) -> np.ndarray:
    """Path index per individual, matching `build_path_alphabet` ordering.

    Paths within a block are enumerated by `product((0,1), repeat=L)`, i.e. the
    first observation is the most significant bit.
    """
    N = y_by_period.shape[1]
    obs = np.zeros((t, N), dtype=np.int64)
    obs[0] = y_by_period[0]
    for s in range(1, t):
        obs[s] = y_by_period[s] - y_by_period[s - 1]
    obs = np.clip(obs, 0, 1)

    off = path_offsets(t)
    idx = np.empty(N, dtype=np.int64)
    exited = exited_by > 0
    # alive through the whole prefix
    alive = ~exited
    if alive.any():
        code = np.zeros(int(alive.sum()), dtype=np.int64)
        for s in range(t):
            code = code * 2 + obs[s][alive]
        idx[alive] = off[None] + code
    for e in range(1, t + 1):
        m = exited_by == e
        if not m.any():
            continue
        code = np.zeros(int(m.sum()), dtype=np.int64)
        for s in range(e):
            code = code * 2 + obs[s][m]
        idx[m] = off[e] + code
    return idx


def empirical_path_counts(data, t: int) -> np.ndarray:
    """Path counts from a simulated prefix (`PrefixData`)."""
    pa = build_path_alphabet(t)
    idx = path_indices(data.y_by_period, data.exited_by, data.active_mask, t)
    return np.bincount(idx, minlength=pa.size).astype(float)


# -- estimate and covariance -------------------------------------------------
def moments_with_covariance(path_counts: np.ndarray, t: int, gamma: int = 1):
    """(mhat, Cov(mhat)) from exact path multinomial counts."""
    pa = build_path_alphabet(t)
    M = cohort_design(t, gamma)
    B = np.linalg.pinv(M) @ pa.to_cohort.T          # (2t, n_paths)
    n = np.asarray(path_counts, dtype=float)
    N = n.sum()
    phat = n / N
    mhat = B @ phat
    cov = (B @ (np.diag(phat) - np.outer(phat, phat)) @ B.T) / N
    return mhat, cov


# -- projection ---------------------------------------------------------------
@dataclass
class ProjectionResult:
    moments: np.ndarray
    was_projected: bool
    cone_statistic_before: float
    cone_statistic_after: float
    weighted_distance: float
    max_abs_moment_shift: float
    max_relative_moment_shift: float
    shifted_moment_index: int
    per_moment_shift: np.ndarray = field(repr=False, default=None)
    n_eigen_kept: int = 0
    weighting: str = "covariance"
    status: str = "ok"


def _whitener(cov: np.ndarray, rcond: float = 1e-10):
    """L with L'L = pinv(cov), from a truncated eigendecomposition."""
    vals, vecs = np.linalg.eigh(cov)
    keep = vals > rcond * max(vals.max(), 1e-300)
    if not np.any(keep):
        return np.eye(cov.shape[0]), 0
    inv_sqrt = np.zeros_like(vals)
    inv_sqrt[keep] = 1.0 / np.sqrt(vals[keep])
    return (vecs * inv_sqrt).T, int(keep.sum())


def project_to_cone(mhat: np.ndarray, cov: np.ndarray, grid: np.ndarray, t: int,
                    weighting: str = "covariance", tol: float = 0.0
                    ) -> ProjectionResult:
    """Covariance-weighted projection of `mhat` onto the [0,1] moment cone."""
    mhat = np.asarray(mhat, dtype=float)
    n_mom = 2 * t
    s_before = cone_statistic(mhat, t)
    if s_before >= -abs(tol):
        return ProjectionResult(mhat.copy(), False, s_before, s_before, 0.0, 0.0,
                                0.0, -1, np.zeros(n_mom), 0, weighting,
                                "already in cone")

    V = np.vstack([grid ** j for j in range(n_mom)])          # (2t, G)
    sub = slice(1, n_mom)                                     # m_0 fixed by sum w = 1
    C = cov[sub, sub]
    if weighting == "covariance":
        L, n_keep = _whitener(C)
    elif weighting == "diagonal":
        d = np.sqrt(np.maximum(np.diag(C), 1e-300))
        L, n_keep = np.diag(1.0 / d), len(d)
    elif weighting == "identity":
        L, n_keep = np.eye(C.shape[0]), C.shape[0]
    else:
        raise ValueError(weighting)

    A = L @ V[sub]
    b = L @ mhat[sub]
    rho = 10.0 * max(1.0, np.abs(A).max())
    A_aug = np.vstack([A, rho * np.ones((1, len(grid)))])
    b_aug = np.concatenate([b, [rho]])
    w, _ = nnls(A_aug, b_aug)
    s = w.sum()
    if s <= 0:
        return ProjectionResult(mhat.copy(), False, s_before, s_before, np.nan,
                                np.nan, np.nan, -1, np.zeros(n_mom), n_keep,
                                weighting, "projection failed: empty nnls solution")
    w = w / s
    m_star = V @ w
    shift = m_star - mhat
    rel = np.abs(shift) / np.maximum(np.abs(mhat), 1e-300)
    return ProjectionResult(
        moments=m_star, was_projected=True,
        cone_statistic_before=s_before,
        cone_statistic_after=cone_statistic(m_star, t),
        weighted_distance=float(np.linalg.norm(L @ shift[sub])),
        max_abs_moment_shift=float(np.max(np.abs(shift))),
        max_relative_moment_shift=float(np.max(rel[1:])),
        shifted_moment_index=int(np.argmax(np.abs(shift))),
        per_moment_shift=shift, n_eigen_kept=n_keep, weighting=weighting)
