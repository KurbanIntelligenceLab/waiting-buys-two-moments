"""Observation model and the cohort/individual observable functionals.

Model (experiment brief, Section 2).  While individual i is active in period t
it emits o_i^t ~ Bernoulli(ptilde_i) and then exits with probability p_i, where

    ptilde = 1 - (1-p)^gamma          (gamma = 1 in the main text)

An individual is *active in period s* iff it survived the exits at the end of
periods 1..s-1, which happens with probability (1-p)^(s-1).  Conditional on
being active in period s it has emitted exactly s observations, so

    phi_{s,k}(p) = P[active in period s, y^s = k]
                 = C(s,k) ptilde^k (1-ptilde)^(s-k) (1-p)^(s-1)

which for gamma = 1 collapses to the brief's

    phi_{s,k}(p) = C(s,k) p^k (1-p)^(s-k) (1-p)^(s-1),      deg = 2s-1.

Everything is represented in the *mixed basis* p^i (1-p)^j.  That choice is not
cosmetic: in this basis every phi has **nonnegative integer** coefficients (because
1 - (1-p)^gamma = p * sum_{r<gamma} (1-p)^r), so expectations are sums of
positive terms and there is no cancellation, at any gamma.  The alternating
monomial representation is produced only where it is actually needed -- the
rank computation of Experiment 1 -- and there it is built in exact integer
arithmetic.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from math import comb

import numpy as np


@dataclass(frozen=True)
class MixedPoly:
    """Polynomial sum_{(i,j)} c_ij * p^i (1-p)^j with integer coefficients."""

    coeffs: tuple  # tuple of ((i, j), c) pairs, canonical sorted order

    @staticmethod
    def from_dict(d: dict) -> "MixedPoly":
        items = {k: int(v) for k, v in d.items() if v != 0}
        return MixedPoly(tuple(sorted(items.items())))

    @staticmethod
    def one() -> "MixedPoly":
        return MixedPoly.from_dict({(0, 0): 1})

    @staticmethod
    def p() -> "MixedPoly":
        return MixedPoly.from_dict({(1, 0): 1})

    @staticmethod
    def one_minus_p() -> "MixedPoly":
        return MixedPoly.from_dict({(0, 1): 1})

    def as_dict(self) -> dict:
        return dict(self.coeffs)

    def __mul__(self, other: "MixedPoly") -> "MixedPoly":
        out: dict = {}
        for (i1, j1), c1 in self.coeffs:
            for (i2, j2), c2 in other.coeffs:
                key = (i1 + i2, j1 + j2)
                out[key] = out.get(key, 0) + c1 * c2
        return MixedPoly.from_dict(out)

    def scale(self, factor: int) -> "MixedPoly":
        return MixedPoly.from_dict({k: c * factor for k, c in self.coeffs})

    def __add__(self, other: "MixedPoly") -> "MixedPoly":
        out = self.as_dict()
        for k, c in other.coeffs:
            out[k] = out.get(k, 0) + c
        return MixedPoly.from_dict(out)

    def power(self, n: int) -> "MixedPoly":
        result = MixedPoly.one()
        base = self
        while n:
            if n & 1:
                result = result * base
            base = base * base
            n >>= 1
        return result

    @property
    def all_nonnegative(self) -> bool:
        return all(c >= 0 for _, c in self.coeffs)

    @property
    def degree(self) -> int:
        return max((i + j for (i, j), _ in self.coeffs), default=0)

    def evaluate(self, p: np.ndarray) -> np.ndarray:
        p = np.asarray(p, dtype=float)
        out = np.zeros_like(p, dtype=float)
        for (i, j), c in self.coeffs:
            out += c * p ** i * (1.0 - p) ** j
        return out

    def expectation(self, dist) -> float:
        """E_dist[poly(p)] -- a sum of positive terms whenever coefficients are."""
        return float(sum(c * dist.mixed(i, j) for (i, j), c in self.coeffs))

    def expectation_exact(self, dist) -> Fraction:
        total = Fraction(0)
        for (i, j), c in self.coeffs:
            total += c * dist.mixed_exact(i, j)
        return total

    def to_monomial(self) -> np.ndarray:
        """Exact integer coefficient vector over p^0 .. p^deg (alternating)."""
        deg = self.degree
        out = [0] * (deg + 1)
        for (i, j), c in self.coeffs:
            for r in range(j + 1):
                out[i + r] += c * comb(j, r) * (-1) ** r
        return np.array(out, dtype=object)


def ptilde_poly(gamma: int) -> MixedPoly:
    """ptilde = 1 - (1-p)^gamma = p * sum_{r=0}^{gamma-1} (1-p)^r."""
    if gamma < 1 or int(gamma) != gamma:
        raise ValueError("gamma must be a positive integer")
    return MixedPoly.from_dict({(1, r): 1 for r in range(int(gamma))})


def one_minus_ptilde_poly(gamma: int) -> MixedPoly:
    """1 - ptilde = (1-p)^gamma."""
    return MixedPoly.from_dict({(0, int(gamma)): 1})


def phi(s: int, k: int, gamma: int = 1) -> MixedPoly:
    """P[active in period s and y^s = k] as a polynomial in p."""
    if not (0 <= k <= s):
        raise ValueError("need 0 <= k <= s")
    term = ptilde_poly(gamma).power(k) * one_minus_ptilde_poly(gamma).power(s - k)
    survive = MixedPoly.from_dict({(0, s - 1): 1})
    return (term * survive).scale(comb(s, k))


def phi_exit(s: int, k: int, gamma: int = 1) -> MixedPoly:
    """P[exits at the end of period s having accumulated y^s = k] = phi_{s,k} * p."""
    return phi(s, k, gamma) * MixedPoly.p()


def phi_survivor(t: int, k: int, gamma: int = 1) -> MixedPoly:
    """P[still active at the *end* of period t with y^t = k] = phi_{t,k} * (1-p).

    Degree 2t, i.e. one more than phi_{t,k}.  This is a well-defined kernel but it
    is NOT part of the length-t observation window: surviving the period-t exit is
    only visible at t+1.  It is kept for the figure scripts and the validation
    suite; `individual_functionals` deliberately does not use it.  See
    results/r15_window_correction_report.md.
    """
    return phi(t, k, gamma) * MixedPoly.one_minus_p()


def cohort_functionals(t: int, gamma: int = 1) -> list:
    """The cohort-count observables through prefix t: {phi_{s,k} : s<=t, k<=s}."""
    return [((s, k), phi(s, k, gamma)) for s in range(1, t + 1) for k in range(s + 1)]


def individual_functionals(t: int, gamma: int = 1) -> list:
    """Individual records through prefix t -- the length-t observation window.

    A record is either `exited at the end of period s` for s < t with y^s = k, or
    `active in period t with y^t = k`.  The period-t exit outcome is *not* in the
    window: a planner watching a length-t prefix cannot see who fails at date t,
    which is visible only at t+1.  So the terminal category is the plain cohort
    kernel phi_{t,k}, not phi_{t,k}*(1-p).

    Every kernel therefore has degree <= 2t-1, the family has exact rank 2t, and
    its span is Pi_{2t-1} -- the same span as the cohort counts, as Lemma 1 of the
    paper states.  The atoms are a genuine probability distribution: summing gives
    [1-(1-p)^(t-1)] + (1-p)^(t-1) = 1.

    Before the R15 correction this enumerated exits through s = t and used
    phi_survivor for the terminal cell, which reaches degree 2t and has rank 2t+1
    -- one moment more than the theory allows.  `record_counts_from_prefix` in
    regret_calibrated.py extracts exactly these atoms and must stay in step with
    this order.  See results/r15_window_correction_report.md.
    """
    out = [(("exit", s, k), phi_exit(s, k, gamma))
           for s in range(1, t) for k in range(s + 1)]
    out += [(("active", t, k), phi(t, k, gamma)) for k in range(t + 1)]
    return out


def utility_poly(s: int, T: int) -> MixedPoly:
    """u^s(p) = 1 - (1-p)^(T-s), the treatment utility at allocation date s.

    Written as p * sum_{r < T-s} (1-p)^r rather than 1 - (1-p)^(T-s) so that the
    coefficients stay nonnegative.  The two forms are algebraically identical,
    but the difference form subtracts two nearly equal expectations when T-s is
    small and loses most of its significant digits exactly where the horizon is
    short and u is itself small.  At s = T this is the zero polynomial.
    """
    if s > T:
        raise ValueError("allocation date cannot exceed the horizon")
    return MixedPoly.from_dict({(1, r): 1 for r in range(T - s)})


def ptilde(p: np.ndarray, gamma: int) -> np.ndarray:
    """Numeric ptilde = 1 - (1-p)^gamma."""
    return 1.0 - (1.0 - np.asarray(p, dtype=float)) ** gamma
