"""Finite-sample confidence identified set (resolves NEEDS_DECISION D8d).

The point-projected set failed for a structural reason: projecting the estimated
moment vector onto the truncated Hausdorff cone lands it on the cone *boundary*,
where the representing measure is unique.  `M_t` then collapses to a single
distribution and "worst case over `M_t`" degenerates into a plug-in.  Measured:
cone statistic after projection ~3e-11 at t = 4, 100% of date pairs separated,
median efficiency 1.0000, and Table 2's "never acts too late" violated on 10% of
runs because the truth is not in the degenerate set.

The fix is to stop pretending the moments are known exactly.  Build

    M_t^alpha = { Q on [0,1] : moments of Q lie in a confidence region for mhat }

which is non-empty whenever the truth is plausible (no projection needed), has
interior by construction (no degeneracy), and contains the truth with probability
at least the nominal coverage -- which is what restores the guarantee in finite
samples.

Shape of the region.  Whiten by the estimated covariance, z = L (m - mhat) with
L'L = Cov(mhat)^+, and take a **box** in the whitened coordinates:

    |z_i| <= r     for i = 1..d,     d = 2t - 1 free moments

The whitened coordinates are asymptotically independent standard normals, so the
simultaneous coverage is exactly (2 Phi(r) - 1)^d.  Setting that equal to
1 - alpha gives the Sidak radius

    r = Phi^{-1}( (1 + (1-alpha)^{1/d}) / 2 )

A box rather than the ellipsoid keeps every constraint linear in the grid
weights, so the whole LP machinery is unchanged: 2d inequality rows instead of
d equality rows.  The box is covariance-weighted through `L`, so a moment
estimated to 1e-6 constrains the set far more tightly than one estimated to 1e-2.

The radius is set by `alpha` alone.  It is never tuned against benchmark
performance, and `nominal_coverage` is recorded on every result.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

ALPHA_DEFAULT = 0.05          # frozen 2026-08-27


@dataclass
class ConfidenceRegion:
    L: np.ndarray             # whitener, (d, d)
    centre: np.ndarray        # mhat restricted to m_1..m_{2t-1}
    radius: float
    alpha: float
    d: int
    nominal_coverage: float
    n_eigen_kept: int
    method: str = "sidak_whitened_box"


def sidak_radius(alpha: float, d: int) -> float:
    """r with (2 Phi(r) - 1)^d = 1 - alpha."""
    per = (1.0 - alpha) ** (1.0 / d)
    return float(norm.ppf(0.5 * (1.0 + per)))


def build_region(mhat: np.ndarray, cov: np.ndarray, t: int,
                 alpha: float = ALPHA_DEFAULT,
                 rcond: float = 1e-10) -> ConfidenceRegion:
    """Whitened-box confidence region for the free moments m_1..m_{2t-1}."""
    mhat = np.asarray(mhat, dtype=float)
    sub = slice(1, 2 * t)                       # m_0 is fixed by sum(w) = 1
    C = np.asarray(cov, dtype=float)[sub, sub]
    d = C.shape[0]

    vals, vecs = np.linalg.eigh(C)
    keep = vals > rcond * max(vals.max(), 1e-300)
    inv_sqrt = np.zeros_like(vals)
    inv_sqrt[keep] = 1.0 / np.sqrt(vals[keep])
    L = (vecs * inv_sqrt).T                     # L'L = pinv(C)

    r = sidak_radius(alpha, d)
    return ConfidenceRegion(
        L=L, centre=mhat[sub], radius=r, alpha=alpha, d=d,
        nominal_coverage=float((2 * norm.cdf(r) - 1) ** d),
        n_eigen_kept=int(keep.sum()))


def box_constraints(region: ConfidenceRegion, grid: np.ndarray, t: int):
    """(A_ub, b_ub) enforcing |L (V w - mhat)|_inf <= r, linear in w."""
    V = np.vstack([grid ** j for j in range(1, 2 * t)])       # (d, G)
    LV = region.L @ V
    Lm = region.L @ region.centre
    A_ub = np.vstack([LV, -LV])
    b_ub = np.concatenate([region.radius + Lm, region.radius - Lm])
    return A_ub, b_ub


def contains(region: ConfidenceRegion, moments: np.ndarray, t: int) -> bool:
    """Does the region contain a given (e.g. the true) moment vector?"""
    m = np.asarray(moments, dtype=float)[1:2 * t]
    return bool(np.max(np.abs(region.L @ (m - region.centre))) <= region.radius)
