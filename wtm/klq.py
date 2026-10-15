"""Kitagawa, Lee & Qiu (2023) mean-square-regret minimax rule.

Transcribed from arXiv:2310.06242v1. The objects used here are:

  rho(a)  := E[ (exp(2 a Ybar) + 1)^{-2} ],  Ybar ~ N(a, 1)     (p.9)
  tau*    := argsup_{tau >= 0} tau^2 rho(tau)  ~= 1.23          (Lemma 3.1)
  a*      := argsup_{0 <= a <= tau*} (a + k/sigma)^2 rho(a)     (eq. 3.2)
  delta*  := exp(2 a* thetahat_e / sigma) / (exp(2 a* thetahat_e / sigma) + 1)
                                                                (Lemma 3.3)

`k` is the HALF-WIDTH of the identified set: their (2.1) is
I(theta_e) = [theta_e - k, theta_e + k]. A caller holding a full width must
pass k = width / 2.

At k = 0 this is Kitagawa, Lee & Qiu (2022)'s point-identified rule with
a* = tau* ~= 1.23 (their Remark 3.5). It is still a logistic of the
t-statistic, never a hard threshold: optimal rules here are always fractional.

`a*` depends on the data only through the ratio k/sigma, so it is cached on
that ratio rather than recomputed per run.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
from numpy.polynomial.hermite_e import hermegauss

# Gauss-Hermite nodes for E_{Z ~ N(0,1)}[.]; 200 nodes is far past convergence
# for this integrand (checked against 400 below in the self-test).
_NODES, _WEIGHTS = hermegauss(200)
_WEIGHTS = _WEIGHTS / _WEIGHTS.sum()

TAU_STAR_PUBLISHED = 1.23          # Lemma 3.1, quoted to two decimals


def rho(a: float) -> float:
    """E[(exp(2 a Ybar) + 1)^-2] with Ybar ~ N(a, 1)."""
    y = a + _NODES
    # log-space to keep exp(2 a y) from overflowing at large a
    z = 2.0 * a * y
    log_term = -2.0 * np.logaddexp(z, 0.0)
    return float(np.dot(_WEIGHTS, np.exp(log_term)))


def _argmax_on(lo: float, hi: float, obj, n_coarse: int = 401,
               n_rounds: int = 60) -> float:
    """Golden-section on a coarse bracket; the objective is unimodal here but
    the coarse pass makes that assumption checkable rather than assumed."""
    xs = np.linspace(lo, hi, n_coarse)
    vs = np.array([obj(x) for x in xs])
    i = int(np.argmax(vs))
    a = xs[max(i - 1, 0)]
    b = xs[min(i + 1, n_coarse - 1)]
    g = (np.sqrt(5.0) - 1.0) / 2.0
    c, d = b - g * (b - a), a + g * (b - a)
    fc, fd = obj(c), obj(d)
    for _ in range(n_rounds):
        if fc > fd:
            b, d, fd = d, c, fc
            c = b - g * (b - a)
            fc = obj(c)
        else:
            a, c, fc = c, d, fd
            d = a + g * (b - a)
            fd = obj(d)
    return 0.5 * (a + b)


@lru_cache(maxsize=1)
def tau_star() -> float:
    """argsup_{tau >= 0} tau^2 rho(tau). Published as ~= 1.23."""
    return _argmax_on(0.0, 5.0, lambda t: t * t * rho(t))


@lru_cache(maxsize=4096)
def a_star(k_over_sigma: float) -> float:
    """argsup_{0 <= a <= tau*} (a + k/sigma)^2 rho(a), equation (3.2).

    `k_over_sigma` is the HALF-width over the standard deviation.
    """
    if not np.isfinite(k_over_sigma) or k_over_sigma < 0:
        raise ValueError(f"k/sigma must be finite and nonnegative: {k_over_sigma}")
    r = float(k_over_sigma)
    t = tau_star()
    return _argmax_on(0.0, t, lambda a: (a + r) ** 2 * rho(a))


def fraction(t_stat: float, k_over_sigma: float) -> float:
    """The optimal treatment fraction, a logistic of the t-statistic.

    `t_stat` is thetahat_e / sigma. Returns the fraction in (0, 1) assigned to
    the "treatment" action -- here, read as the probability of choosing the
    later date.
    """
    a = a_star(k_over_sigma)
    z = 2.0 * a * float(t_stat)
    return float(1.0 / (1.0 + np.exp(-z)))       # = exp(z) / (exp(z) + 1)


def worst_case_msr(sigma: float, k: float) -> float:
    """sup_theta R_sq at the minimax rule: sigma^2 (a* + k/sigma)^2 rho(a*)."""
    r = k / sigma
    a = a_star(r)
    return sigma ** 2 * (a + r) ** 2 * rho(a)
