"""Finite-sample simulation of a planner's observed prefix.

Draw order is fixed and documented so that a (config, seed) pair reproduces
bit-for-bit:

    1. p_1..p_N                        risks, one draw block
    2. for s = 1..t:  o^s for the currently active individuals
    3. for s = 1..t:  exit indicators for the currently active individuals

Only periods 1..t are simulated.  The planner observes nothing after the prefix,
and selected dates are scored against the *mean-field* trajectory, so simulating
beyond t would add cost and no information.

One `PrefixData` is built per (configuration, seed) and handed to every timing
rule unchanged.  Rules must never draw their own data; a rule that needs
randomness gets a separate, explicitly recorded rule seed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .observation import ptilde


@dataclass(frozen=True)
class PrefixData:
    """What the planner sees after t periods, plus hidden truth for diagnostics."""

    N: int
    t: int
    T: int
    gamma: int
    beta: float
    seed: int

    y: np.ndarray            # (N,) observation count at the individual's last active period
    active_in: np.ndarray    # (N,) last period the individual was active (1..t)
    exited_by: np.ndarray    # (N,) period at whose end the individual exited, or -1
    y_by_period: np.ndarray  # (t, N) running count y^s, valid where active
    active_mask: np.ndarray  # (t, N) True where individual was active in period s
    p_true: np.ndarray = field(repr=False, default=None)   # hidden from rules

    # -- planner-observable summaries ---------------------------------------
    def cohort_counts(self) -> np.ndarray:
        """C[s-1, k] = number of individuals active in period s with y^s = k."""
        C = np.zeros((self.t, self.t + 1), dtype=np.int64)
        for s in range(1, self.t + 1):
            m = self.active_mask[s - 1]
            if m.any():
                C[s - 1] = np.bincount(self.y_by_period[s - 1][m].astype(np.int64),
                                       minlength=self.t + 1)[:self.t + 1]
        return C

    def individual_records(self):
        """(exited_by, y_at_last_active) -- the atoms of the single-record law."""
        return self.exited_by.copy(), self.y.copy()

    def empirical_active_fraction(self) -> np.ndarray:
        return self.active_mask.mean(axis=1)


def simulate_prefix(dist, N: int, T: int, gamma: int, t: int, seed: int,
                    keep_truth: bool = True) -> PrefixData:
    rng = np.random.default_rng(int(seed))
    N = int(N)
    p = dist.sample(rng, N)
    pt = ptilde(p, gamma)

    active = np.ones(N, dtype=bool)
    y = np.zeros(N, dtype=np.int32)
    y_by_period = np.zeros((t, N), dtype=np.int32)
    active_mask = np.zeros((t, N), dtype=bool)
    exited_by = np.full(N, -1, dtype=np.int32)
    last_active = np.zeros(N, dtype=np.int32)

    for s in range(1, t + 1):
        active_mask[s - 1] = active
        if not active.any():
            break
        obs = rng.random(N) < pt          # drawn for all, used only where active
        y[active] += obs[active]
        y_by_period[s - 1] = y
        last_active[active] = s
        exits = (rng.random(N) < p) & active
        exited_by[exits] = s
        active = active & ~exits

    return PrefixData(N=N, t=t, T=T, gamma=gamma, beta=0.0, seed=int(seed),
                      y=y, active_in=last_active, exited_by=exited_by,
                      y_by_period=y_by_period, active_mask=active_mask,
                      p_true=p if keep_truth else None)


def empirical_cohort_moments(data: PrefixData) -> np.ndarray:
    """Normalised cohort counts: phihat_{s,k} = C[s,k] / N, the sample analogue
    of phi_{s,k} = E[phi_{s,k}(p)]."""
    return data.cohort_counts() / data.N


# -- model departures for the specification test (Experiments 22-23) ----------
# Both mechanisms are frozen in experiments/reports/SPEC_TEST_MECHANISMS.md and
# must not be changed now that results exist.  Each returns cohort counts only,
# which is all the moment-cone diagnostic consumes.

def _cohort_counts_from_paths(y_by_period, active_mask, t):
    C = np.zeros((t, t + 1), dtype=np.int64)
    for s in range(1, t + 1):
        m = active_mask[s - 1]
        if m.any():
            C[s - 1] = np.bincount(y_by_period[s - 1][m].astype(np.int64),
                                   minlength=t + 1)[:t + 1]
    return C


def simulate_cohorts_drift(dist, N, t, seed, delta):
    """Experiment 22: risk escalates geometrically, p_i^s = min(p_i (1+delta)^(s-1), 1).

    delta = 0 reproduces the null model exactly.  The drifted probability governs
    both the flag and the exit in period s (ptilde = p at gamma = 1).
    """
    rng = np.random.default_rng(int(seed))
    p0 = dist.sample(rng, int(N))
    active = np.ones(int(N), dtype=bool)
    y = np.zeros(int(N), dtype=np.int32)
    y_by_period = np.zeros((t, int(N)), dtype=np.int32)
    active_mask = np.zeros((t, int(N)), dtype=bool)
    for s in range(1, t + 1):
        active_mask[s - 1] = active
        p_s = np.minimum(p0 * (1.0 + delta) ** (s - 1), 1.0)
        obs = rng.random(int(N)) < p_s
        y[active] += obs[active]
        y_by_period[s - 1] = y
        active &= ~((rng.random(int(N)) < p_s) & active)
    return _cohort_counts_from_paths(y_by_period, active_mask, t)


def simulate_cohorts_group_correlated(dist, N, t, seed, group_size, rho):
    """Experiment 23: failures correlated within groups, marginals preserved.

    Each group draws one shared uniform per period.  An individual uses the
    shared draw with probability rho and its own otherwise, so its marginal
    failure probability is exactly p_i -- only the dependence changes.  Flags
    stay independent across individuals.
    """
    rng = np.random.default_rng(int(seed))
    N = int(N)
    p = dist.sample(rng, N)
    group = np.arange(N) // int(group_size)
    n_groups = int(group.max()) + 1
    active = np.ones(N, dtype=bool)
    y = np.zeros(N, dtype=np.int32)
    y_by_period = np.zeros((t, N), dtype=np.int32)
    active_mask = np.zeros((t, N), dtype=bool)
    for s in range(1, t + 1):
        active_mask[s - 1] = active
        obs = rng.random(N) < p
        y[active] += obs[active]
        y_by_period[s - 1] = y
        shared = rng.random(n_groups)[group]
        own = rng.random(N)
        use_shared = rng.random(N) < rho
        V = np.where(use_shared, shared, own)
        active &= ~((V < p) & active)
    return _cohort_counts_from_paths(y_by_period, active_mask, t)


def simulate_cohorts_null(dist, N, t, seed):
    """The correctly specified model, returned as cohort counts."""
    return simulate_cohorts_drift(dist, N, t, seed, 0.0)
