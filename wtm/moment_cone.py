"""Truncated Hausdorff moment-cone feasibility test on [0,1].

A length-t prefix identifies m_0..m_{2t-1}: an odd number of steps,
n = 2t-1 = 2k+1 with k = t-1.  For that case a representing measure on [0,1]
exists iff both localizing matrices are positive semidefinite:

    A_k = ( m_{i+j+1} )_{i,j=0..k}                localizing for p >= 0
    C_k = ( m_{i+j} - m_{i+j+1} )_{i,j=0..k}      localizing for 1-p >= 0

Both are required.  Hankel positivity alone is the condition for the real line
and is *not* sufficient here -- the draft's own counterexample (1,2,4,8,16) is
Hankel PSD, being the moments of a point mass at p = 2, yet has no representing
measure on [0,1].  `test_moment_cone.py` checks that this implementation rejects
it.

Paths.  For the bootstrap, the i.i.d. unit is an individual's full observation
path through the prefix, not the (exit, count) pair: cohort counts depend on
*when* the flags arrived, which the pair does not determine.  There are only
sum_{e<=t} 2^e + 2^t distinct paths (46 at t = 4), so the whole sampling model
is one multinomial over a small alphabet plus a fixed 0/1 matrix mapping paths
to cohort counts.  That makes a bootstrap replicate a single multinomial draw
and a matrix product, rather than a resimulation of N individuals.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np
from scipy.optimize import nnls

from .observation import cohort_functionals


def cone_matrices(m: np.ndarray):
    """The two PSD matrices whose positivity characterises the [0,1] cone.

    For moments m_0..m_n the Karlin-Studden conditions depend on the parity of n:

        n = 2k   :  H_k     = (m_{i+j})_{i,j=0..k}                  >= 0
                    B_{k-1} = (m_{i+j+1} - m_{i+j+2})_{i,j=0..k-1}  >= 0
        n = 2k+1 :  A_k     = (m_{i+j+1})_{i,j=0..k}                >= 0
                    C_k     = (m_{i+j} - m_{i+j+1})_{i,j=0..k}      >= 0

    A prefix of length t supplies m_0..m_{2t-1}, always the odd case.  The even
    case is implemented too, because the draft's counterexample (1,2,4,8,16) has
    an even number of steps and is the check that this is a genuine [0,1] cone
    test rather than Hankel positivity alone.
    """
    m = np.asarray(m, dtype=float)
    n = len(m) - 1
    if n < 1:
        raise ValueError("need at least m_0 and m_1")
    if n % 2 == 0:
        k = n // 2
        H = np.array([[m[i + j] for j in range(k + 1)] for i in range(k + 1)])
        if k == 0:
            return H, np.array([[m[0]]])
        B = np.array([[m[i + j + 1] - m[i + j + 2] for j in range(k)]
                      for i in range(k)])
        return H, B
    k = (n - 1) // 2
    A = np.array([[m[i + j + 1] for j in range(k + 1)] for i in range(k + 1)])
    C = np.array([[m[i + j] - m[i + j + 1] for j in range(k + 1)]
                  for i in range(k + 1)])
    return A, C


def localizing_matrices(m: np.ndarray, t: int):
    """The odd-case pair (A_k, C_k) for a length-t prefix, k = t-1."""
    m = np.asarray(m, dtype=float)
    need = 2 * t
    if len(m) < need:
        raise ValueError(f"need {need} moments for t={t}, got {len(m)}")
    return cone_matrices(m[:need])


def cone_statistic(m: np.ndarray, t: int = None) -> float:
    """S = min over both cone matrices of trace-normalised lambda_min.

    S > 0 means the moment vector lies strictly inside the cone.  Each matrix is
    scaled by its own trace so the statistic does not inherit the overall
    magnitude of the moments, which varies by orders of magnitude across the
    risk distributions.  Pass `t` to use exactly the first 2t moments.
    """
    A, C = localizing_matrices(m, t) if t is not None else cone_matrices(m)
    out = []
    for M in (A, C):
        tr = np.trace(M)
        scale = abs(tr) if abs(tr) > 0 else 1.0
        out.append(float(np.linalg.eigvalsh(M)[0]) / scale)
    return float(min(out))


def in_cone(m: np.ndarray, t: int = None, tol: float = 0.0) -> bool:
    return cone_statistic(m, t) >= -abs(tol)


# -- observable -> moments ----------------------------------------------------
def cohort_design(t: int, gamma: int = 1):
    """Matrix M with n = M m, rows ordered as `cohort_functionals(t, gamma)`."""
    funcs = cohort_functionals(t, gamma)
    width = 2 * t
    M = np.zeros((len(funcs), width))
    for r, (_, poly) in enumerate(funcs):
        v = np.array(poly.to_monomial(), dtype=float)
        M[r, :len(v)] = v[:width]
    return M


def moments_from_cohort(n_flat: np.ndarray, t: int, gamma: int = 1) -> np.ndarray:
    """Recover m_0..m_{2t-1} from cohort proportions (least squares)."""
    M = cohort_design(t, gamma)
    return np.linalg.pinv(M) @ np.asarray(n_flat, dtype=float)


def flatten_cohort_counts(C: np.ndarray, t: int) -> np.ndarray:
    """Flatten a (t, t+1) cohort-count array into `cohort_functionals` order."""
    return np.array([C[s - 1, k] for s in range(1, t + 1) for k in range(s + 1)],
                    dtype=float)


# -- path alphabet ------------------------------------------------------------
@dataclass(frozen=True)
class PathAlphabet:
    """All distinct observation paths through a prefix of length t."""

    t: int
    paths: tuple            # (exit_period or None, observation tuple)
    to_cohort: np.ndarray   # (n_paths, n_cohort_cells) 0/1 incidence
    exps: np.ndarray        # (n_paths, 2) exponents (i, j) of p^i (1-p)^j

    @property
    def size(self) -> int:
        return len(self.paths)

    def probabilities(self, dist) -> np.ndarray:
        q = np.array([dist.mixed(int(i), int(j)) for i, j in self.exps])
        return q / q.sum()

    def probabilities_from_grid(self, grid: np.ndarray, w: np.ndarray) -> np.ndarray:
        q = np.array([np.sum(w * grid ** int(i) * (1.0 - grid) ** int(j))
                      for i, j in self.exps])
        s = q.sum()
        return q / s if s > 0 else q


def build_path_alphabet(t: int) -> PathAlphabet:
    paths, rows, exps = [], [], []
    cells = [(s, k) for s in range(1, t + 1) for k in range(s + 1)]
    index = {c: i for i, c in enumerate(cells)}

    def add(exit_period, obs):
        row = np.zeros(len(cells))
        run = 0
        active_through = len(obs)
        for s in range(1, active_through + 1):
            run += obs[s - 1]
            row[index[(s, run)]] = 1.0
        k = int(sum(obs))
        if exit_period is None:                  # alive at end of t
            i, j = k, 2 * t - k
        else:                                    # exits at end of `exit_period`
            e = exit_period
            i, j = k + 1, 2 * e - 1 - k
        paths.append((exit_period, tuple(obs)))
        rows.append(row)
        exps.append((i, j))

    for e in range(1, t + 1):
        for obs in product((0, 1), repeat=e):
            add(e, obs)
    for obs in product((0, 1), repeat=t):
        add(None, obs)
    return PathAlphabet(t=t, paths=tuple(paths), to_cohort=np.array(rows),
                        exps=np.array(exps))


# -- null projection ----------------------------------------------------------
def fit_null_on_grid(n_flat: np.ndarray, t: int, grid: np.ndarray,
                     gamma: int = 1) -> np.ndarray:
    """NNLS fit of a grid distribution to observed cohort proportions.

    Any distribution supported on the grid automatically has moments inside the
    cone, so its record law is a bona fide null from which to bootstrap.
    """
    from .observation import phi
    Phi = np.vstack([phi(s, k, gamma).evaluate(grid)
                     for s in range(1, t + 1) for k in range(s + 1)])
    # append the normalisation row with a large weight so sum(w) ~ 1
    big = 10.0 * max(1.0, np.abs(Phi).max())
    A = np.vstack([Phi, big * np.ones(len(grid))])
    b = np.concatenate([np.asarray(n_flat, float), [big]])
    w, _ = nnls(A, b)
    s = w.sum()
    return w / s if s > 0 else np.full(len(grid), 1.0 / len(grid))
