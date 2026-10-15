"""Gauss quadrature rules used to build moment-matched atomic twins.

The construction answers one question: given a smooth risk distribution P, is
there an *atomic* distribution that a length-tau prefix cannot tell apart from
it?  Lemma `lem:moments` says the prefix sees P only through m_0..m_{2tau-1},
so any distribution matching that many moments generates the same record law.

The n-point Gauss rule with respect to P is exactly such a distribution: it has
n atoms and reproduces E[p^j] for j <= 2n-1.  Its nodes are the roots of the
degree-n polynomial orthogonal to Pi_{n-1} under P, and its weights are
positive whenever P has n points of increase -- so the rule is a genuine
probability distribution on [0,1], not a signed one.

Everything up to root-finding is done in exact rational arithmetic, because the
Hankel systems here are severely ill-conditioned (for Beta(1,32), m_9 is of
order 1e-11 and the degree-5 Hankel determinant underflows double precision).
Roots and weights are then solved in `mpmath` at high precision and only
rounded at the very end.
"""

from fractions import Fraction

import mpmath as mp
import numpy as np

from .distributions import DiscreteRisk

__all__ = ["raw_moments_exact", "gauss_rule", "matched_atomic"]


def raw_moments_exact(dist, J: int) -> list:
    """[m_0, ..., m_J] as exact Fractions.  Requires `dist.exact_available`."""
    if not dist.exact_available:
        raise ValueError(f"{dist!r} has no exact moments; refusing to build a "
                         "matched pair from floating-point moments")
    return [Fraction(dist.mixed_exact(j, 0)) for j in range(J + 1)]


def _solve_exact(A: list, b: list) -> list:
    """Gaussian elimination with partial pivoting over Fraction, no rounding."""
    n = len(b)
    M = [list(row) + [rhs] for row, rhs in zip(A, b)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(M[r][col]))
        if M[piv][col] == 0:
            raise np.linalg.LinAlgError("singular moment (Hankel) system")
        M[col], M[piv] = M[piv], M[col]
        inv = Fraction(1, 1) / M[col][col]
        M[col] = [v * inv for v in M[col]]
        for r in range(n):
            if r != col and M[r][col] != 0:
                f = M[r][col]
                M[r] = [v - f * w for v, w in zip(M[r], M[col])]
    return [M[r][n] for r in range(n)]


def gauss_rule(moments: list, n: int, dps: int = 120):
    """n-point Gauss rule from exact moments m_0..m_{2n-1}.

    Returns (nodes, weights) as floats.  Exact for every polynomial of degree
    <= 2n-1, which is the property the matched pair is built on.
    """
    if len(moments) < 2 * n:
        raise ValueError(f"need {2*n} moments for an {n}-point rule, "
                         f"got {len(moments)}")
    m = [Fraction(x) for x in moments]

    # Monic orthogonal polynomial pi_n(x) = x^n + sum_j c_j x^j, defined by
    # <pi_n, x^k> = 0 for k < n:  sum_j c_j m_{k+j} = -m_{k+n}.
    A = [[m[k + j] for j in range(n)] for k in range(n)]
    b = [-m[k + n] for k in range(n)]
    c = _solve_exact(A, b)

    with mp.workdps(dps):
        coeffs = [mp.mpf(1)] + [mp.mpf(x.numerator) / mp.mpf(x.denominator)
                                for x in reversed(c)]
        roots = mp.polyroots(coeffs, maxsteps=200, extraprec=400)
        nodes = sorted(mp.re(r) for r in roots)

        # Weights from the Vandermonde moment conditions sum_i w_i x_i^k = m_k.
        V = mp.matrix(n, n)
        for k in range(n):
            for i in range(n):
                V[k, i] = nodes[i] ** k
        rhs = mp.matrix([mp.mpf(m[k].numerator) / mp.mpf(m[k].denominator)
                         for k in range(n)])
        w = mp.lu_solve(V, rhs)
        nodes_f = [float(x) for x in nodes]
        weights_f = [float(w[i]) for i in range(n)]

    return np.array(nodes_f), np.array(weights_f)


def matched_atomic(dist, n: int = 5, name: str = None) -> DiscreteRisk:
    """The n-atom twin of `dist`: same m_0..m_{2n-1}, different geometry.

    Raises if the rule leaves [0,1] or picks up a nonpositive weight, which
    would mean the Hankel system was solved outside its regularity conditions
    and the "twin" is not a probability distribution.
    """
    moments = raw_moments_exact(dist, 2 * n - 1)
    nodes, weights = gauss_rule(moments, n)

    if np.any(nodes <= 0) or np.any(nodes >= 1):
        raise ValueError(f"Gauss nodes outside (0,1): {nodes}")
    if np.any(weights <= 0):
        raise ValueError(f"nonpositive Gauss weight: {weights}")
    weights = weights / weights.sum()

    base = getattr(dist, "name", "dist")
    return DiscreteRisk(points=tuple(float(x) for x in nodes),
                        weights=tuple(float(x) for x in weights),
                        name=name or f"atomic{n}_{base}")
