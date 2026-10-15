"""Risk distributions P^1 supported on [0, 1].

The whole mean-field machinery reduces to two moment primitives:

    q(m)        = E[(1-p)^m]
    mixed(i, j) = E[p^i (1-p)^j]

Both are exposed in float and -- where the distribution admits it -- in exact
rational arithmetic.  The three Beta families the paper's figures use are all
Beta(1, b) with integer b, so the exact path covers the entire frozen benchmark
and lets the validation suite check floating point against an exact reference.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from fractions import Fraction

import numpy as np
from scipy.special import betainc, betaln
from scipy.stats import beta as _beta_rv


class RiskDistribution:
    """Interface every risk distribution implements."""

    name: str

    def mixed(self, i: int, j: int) -> float:
        """E[p^i (1-p)^j]."""
        raise NotImplementedError

    def q(self, m: int) -> float:
        """E[(1-p)^m]."""
        return self.mixed(0, m)

    def raw_moment(self, j: int) -> float:
        """E[p^j]."""
        return self.mixed(j, 0)

    # -- exact rational path (optional) --------------------------------------
    @property
    def exact_available(self) -> bool:
        return False

    def mixed_exact(self, i: int, j: int) -> Fraction:
        raise NotImplementedError(f"{self.name} has no exact rational moments")

    def q_exact(self, m: int) -> Fraction:
        return self.mixed_exact(0, m)

    def tail_mixed(self, i: int, j: int, x: float) -> float:
        """E[p^i (1-p)^j 1{p >= x}] -- needed for full-information ranking."""
        raise NotImplementedError

    # -- sampling / cdf ------------------------------------------------------
    def sample(self, rng: np.random.Generator, n: int) -> np.ndarray:
        raise NotImplementedError

    def cdf(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def describe(self) -> dict:
        raise NotImplementedError


def _beta_fn_exact(a: int, b: int) -> Fraction:
    """B(a, b) for positive integers a, b  =  (a-1)!(b-1)!/(a+b-1)!."""
    return Fraction(math.factorial(a - 1) * math.factorial(b - 1),
                    math.factorial(a + b - 1))


def _fmt(x: float) -> str:
    return str(int(x)) if float(x).is_integer() else str(x).replace(".", "p")


@dataclass(frozen=True)
class BetaRisk(RiskDistribution):
    """Beta(a, b) risk distribution."""

    a: float
    b: float
    name: str = field(default="")

    def __post_init__(self):
        if self.a <= 0 or self.b <= 0:
            raise ValueError("Beta parameters must be positive")
        if not self.name:
            object.__setattr__(self, "name", f"beta_{_fmt(self.a)}_{_fmt(self.b)}")

    def mixed(self, i: int, j: int) -> float:
        return float(np.exp(betaln(self.a + i, self.b + j) - betaln(self.a, self.b)))

    @property
    def exact_available(self) -> bool:
        return float(self.a).is_integer() and float(self.b).is_integer()

    def mixed_exact(self, i: int, j: int) -> Fraction:
        if not self.exact_available:
            raise NotImplementedError("exact moments need integer Beta parameters")
        a, b = int(self.a), int(self.b)
        return _beta_fn_exact(a + i, b + j) / _beta_fn_exact(a, b)

    def tail_mixed(self, i: int, j: int, x: float) -> float:
        x = float(np.clip(x, 0.0, 1.0))
        return self.mixed(i, j) * float(1.0 - betainc(self.a + i, self.b + j, x))

    def sample(self, rng: np.random.Generator, n: int) -> np.ndarray:
        return rng.beta(self.a, self.b, size=n)

    def cdf(self, x: np.ndarray) -> np.ndarray:
        return _beta_rv.cdf(x, self.a, self.b)

    def pdf(self, x: np.ndarray) -> np.ndarray:
        return _beta_rv.pdf(x, self.a, self.b)

    def describe(self) -> dict:
        return {"family": "beta", "a": self.a, "b": self.b, "name": self.name,
                "mean": self.a / (self.a + self.b)}


@dataclass(frozen=True)
class MixtureRisk(RiskDistribution):
    """Finite mixture of risk distributions."""

    components: tuple
    weights: tuple
    name: str = field(default="")

    def __post_init__(self):
        w = np.asarray(self.weights, dtype=float)
        if len(w) != len(self.components):
            raise ValueError("weights and components must have equal length")
        if np.any(w < 0) or not math.isclose(w.sum(), 1.0, rel_tol=0, abs_tol=1e-12):
            raise ValueError("mixture weights must be nonnegative and sum to 1")
        if not self.name:
            object.__setattr__(
                self, "name", "mixture_" + "_".join(c.name for c in self.components))

    def mixed(self, i: int, j: int) -> float:
        return float(sum(w * c.mixed(i, j)
                         for w, c in zip(self.weights, self.components)))

    @property
    def exact_available(self) -> bool:
        return all(c.exact_available for c in self.components)

    def mixed_exact(self, i: int, j: int) -> Fraction:
        total = Fraction(0)
        for w, c in zip(self.weights, self.components):
            total += Fraction(w).limit_denominator(10 ** 9) * c.mixed_exact(i, j)
        return total

    def tail_mixed(self, i: int, j: int, x: float) -> float:
        return float(sum(w * c.tail_mixed(i, j, x)
                         for w, c in zip(self.weights, self.components)))

    def sample(self, rng: np.random.Generator, n: int) -> np.ndarray:
        which = rng.choice(len(self.components), size=n, p=np.asarray(self.weights))
        out = np.empty(n, dtype=float)
        for idx, comp in enumerate(self.components):
            mask = which == idx
            k = int(mask.sum())
            if k:
                out[mask] = comp.sample(rng, k)
        return out

    def cdf(self, x: np.ndarray) -> np.ndarray:
        return sum(w * c.cdf(x) for w, c in zip(self.weights, self.components))

    def describe(self) -> dict:
        return {"family": "mixture", "name": self.name,
                "weights": list(self.weights),
                "components": [c.describe() for c in self.components]}


@dataclass(frozen=True)
class DiscreteRisk(RiskDistribution):
    """Atomic distribution: mass `weights` at support points `points`."""

    points: tuple
    weights: tuple
    name: str = field(default="discrete")

    def __post_init__(self):
        p = np.asarray(self.points, dtype=float)
        w = np.asarray(self.weights, dtype=float)
        if p.shape != w.shape:
            raise ValueError("points and weights must have the same shape")
        if np.any(p < 0) or np.any(p > 1):
            raise ValueError("support must lie in [0, 1]")
        if np.any(w < 0) or not math.isclose(w.sum(), 1.0, rel_tol=0, abs_tol=1e-10):
            raise ValueError("weights must be nonnegative and sum to 1")

    def mixed(self, i: int, j: int) -> float:
        p = np.asarray(self.points, dtype=float)
        w = np.asarray(self.weights, dtype=float)
        return float(np.sum(w * p ** i * (1.0 - p) ** j))

    def tail_mixed(self, i: int, j: int, x: float) -> float:
        p = np.asarray(self.points, dtype=float)
        w = np.asarray(self.weights, dtype=float)
        m = p >= x
        return float(np.sum(w[m] * p[m] ** i * (1.0 - p[m]) ** j))

    def sample(self, rng: np.random.Generator, n: int) -> np.ndarray:
        idx = rng.choice(len(self.points), size=n, p=np.asarray(self.weights))
        return np.asarray(self.points, dtype=float)[idx]

    def cdf(self, x: np.ndarray) -> np.ndarray:
        p = np.asarray(self.points, dtype=float)
        w = np.asarray(self.weights, dtype=float)
        x = np.atleast_1d(np.asarray(x, dtype=float))
        return np.array([w[p <= xi].sum() for xi in x])

    def describe(self) -> dict:
        return {"family": "discrete", "name": self.name,
                "points": list(self.points), "weights": list(self.weights)}


def from_spec(spec: dict) -> RiskDistribution:
    """Build a distribution from a plain dict (as stored in the frozen YAML)."""
    fam = spec["family"]
    if fam == "beta":
        return BetaRisk(a=float(spec["a"]), b=float(spec["b"]),
                        name=spec.get("name", ""))
    if fam == "mixture":
        comps = tuple(from_spec(c) for c in spec["components"])
        return MixtureRisk(components=comps,
                           weights=tuple(float(w) for w in spec["weights"]),
                           name=spec.get("name", ""))
    if fam == "discrete":
        return DiscreteRisk(points=tuple(float(x) for x in spec["points"]),
                            weights=tuple(float(w) for w in spec["weights"]),
                            name=spec.get("name", "discrete"))
    raise ValueError(f"unknown distribution family: {fam!r}")
