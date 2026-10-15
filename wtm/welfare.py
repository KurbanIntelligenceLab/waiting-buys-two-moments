"""Mean-field welfare: W^s (ranking by y^s), Wbar^s (full information), t*.

All quantities here are *population* quantities computed from the distribution's
exact moments.  They are the ground truth against which finite-sample rules are
scored; the brief forbids using a finite sample's own realised welfare as the
target.

Allocation.  At date s the planner has budget ratio beta = B/N and ranks the
individuals active in period s by y^s, breaking ties uniformly at random.  Since
the observation kernel is MLR in p at every gamma, treating whole groups from
the top y-group downwards is the optimal y-measurable allocation, so

    W^s = max_a { sum_k a_k v_{s,k} : sum_k a_k m_{s,k} <= beta, a_k in [0,1] }

is attained by the greedy fill implemented in `fill_greedy`.  Writing W^s as the
value of this small LP is what makes the identified-set bounds of Experiment 4
tractable, so the same primitive is reused there.

    m_{s,k} = E[phi_{s,k}(p)]                 mass of active individuals with y^s = k
    v_{s,k} = E[phi_{s,k}(p) u^s(p)]          their utility mass
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy.optimize import brentq

from .observation import phi, utility_poly


def group_polys(s: int, gamma: int, T: int):
    """(mass polynomial, utility polynomial) for each y-group k = 0..s."""
    u = utility_poly(s, T)
    mass = [phi(s, k, gamma) for k in range(s + 1)]
    util = [m * u for m in mass]
    return mass, util


def group_masses(dist, s: int, gamma: int) -> np.ndarray:
    """m_{s,k} for k = 0..s."""
    return np.array([phi(s, k, gamma).expectation(dist) for k in range(s + 1)])


def group_utilities(dist, s: int, gamma: int, T: int) -> np.ndarray:
    """v_{s,k} for k = 0..s."""
    u = utility_poly(s, T)
    return np.array([(phi(s, k, gamma) * u).expectation(dist) for k in range(s + 1)])


def fill_greedy(masses: np.ndarray, utils: np.ndarray, beta: float):
    """Greedy top-down budget fill.  Returns (welfare, fill fractions a_k)."""
    masses = np.asarray(masses, dtype=float)
    utils = np.asarray(utils, dtype=float)
    a = np.zeros_like(masses)
    remaining = float(beta)
    for k in range(len(masses) - 1, -1, -1):
        if remaining <= 0.0:
            break
        m = masses[k]
        if m <= 0.0:
            continue
        take = min(remaining, m)
        a[k] = take / m
        remaining -= take
    return float(np.dot(a, utils)), a


def welfare_y(dist, s: int, gamma: int, T: int, beta: float) -> float:
    """W^s -- welfare from ranking the active pool by y^s."""
    m = group_masses(dist, s, gamma)
    v = group_utilities(dist, s, gamma, T)
    return fill_greedy(m, v, beta)[0]


def allocation_profile(dist, s: int, gamma: int, T: int, beta: float) -> np.ndarray:
    """The fill fractions a_k that attain W^s."""
    m = group_masses(dist, s, gamma)
    v = group_utilities(dist, s, gamma, T)
    return fill_greedy(m, v, beta)[1]


def risk_threshold(dist, s: int, beta: float) -> float:
    """p* with E[(1-p)^(s-1) 1{p >= p*}] = beta; 0 if the whole pool fits."""
    active = dist.tail_mixed(0, s - 1, 0.0)
    if active <= beta:
        return 0.0

    def g(x):
        return dist.tail_mixed(0, s - 1, x) - beta

    if g(1.0) > 0:          # an atom at p = 1 already exceeds the budget
        return 1.0
    return float(brentq(g, 0.0, 1.0, xtol=1e-14, rtol=1e-15, maxiter=500))


def welfare_full_info(dist, s: int, T: int, beta: float) -> float:
    """Wbar^s -- welfare when the planner ranks the active pool by the true p.

    Independent of gamma: the observation channel is irrelevant once p is known,
    and the active pool (1-p)^(s-1) does not depend on gamma either.
    """
    pstar = risk_threshold(dist, s, beta)
    # E[(1-p)^(s-1) u^s(p) 1{p >= p*}] with u^s = p * sum_{r < T-s} (1-p)^r
    return float(sum(dist.tail_mixed(1, s - 1 + r, pstar) for r in range(T - s)))


@dataclass(frozen=True)
class PopulationConfig:
    """A population-level cell: everything needed to pin down W^1..W^T."""

    dist: object
    gamma: int
    T: int
    beta: float

    @property
    def dates(self) -> np.ndarray:
        """Candidate allocation dates.  s = T is excluded: u^T == 0 there."""
        return np.arange(1, self.T)

    @lru_cache(maxsize=None)
    def _traj(self):
        w = np.array([welfare_y(self.dist, int(s), self.gamma, self.T, self.beta)
                      for s in self.dates])
        wbar = np.array([welfare_full_info(self.dist, int(s), self.T, self.beta)
                         for s in self.dates])
        return w, wbar

    @property
    def W(self) -> np.ndarray:
        return self._traj()[0]

    @property
    def Wbar(self) -> np.ndarray:
        return self._traj()[1]

    @property
    def t_star(self) -> int:
        return int(self.dates[int(np.argmax(self.W))])

    @property
    def W_star(self) -> float:
        return float(np.max(self.W))

    def welfare_at(self, s: int) -> float:
        return float(self.W[int(s) - 1])

    def efficiency(self, s: int) -> float:
        """Welfare efficiency W^s / W^{t*} of acting at date s."""
        return self.welfare_at(s) / self.W_star

    def relative_regret(self, s: int) -> float:
        """R(s) = 1 - W^s / W^{t*}."""
        return 1.0 - self.efficiency(s)

    @property
    def stakes(self) -> float:
        """Relative welfare loss from acting immediately rather than optimally.

        stakes = 1 - W^1 / W^{t*}.  This is `relative_regret(1)`; it is the
        Experiment 2 quantity and uses the same relative-regret convention as
        every other regret number in this codebase.
        """
        return self.relative_regret(1)
