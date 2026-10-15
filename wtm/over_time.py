"""Shirali, Procaccia & Abebe (ICLR 2025) Algorithm 1: optimal over-time allocation.

Ported from their MATLAB reference implementation
(find_optimal_over_time_alloc.m, simulate_traj.m, posterior_at_k_t.m,
value_at_k_t.m) at github.com/alishiraliGit/hidden-costs-of-waiting-for-accurate-predictions.

WHY A PORT RATHER THAN THEIR CODE.  Two reasons, both about making the
comparison meaningful rather than about convenience.

1. TIMING CONVENTION.  Their posterior_at_k_t(k, t) returns Beta(a0 + k,
   b0 + 2t - k), so by their time t an individual has taken t signal draws AND
   t survival draws.  Our phi_{s,k} carries (1-p)^(2s-k-1): s signal draws and
   s-1 survival draws.  The models order the within-period signal and failure
   draws differently around the allocation point -- ours allocates after the
   period's signal but before its failure, theirs after both.  Running their
   code unchanged would compare two different populations, so the survival
   exponent is a parameter here (`offset`) and both conventions are available.

2. THEIR IMPLEMENTATION IS BETA-ONLY.  It propagates a conjugate Beta
   posterior in closed form, which cannot express our two mixture families.
   This port carries the same quantities as ratios of exact mixed moments
   E[p^i (1-p)^j], so it runs on any distribution our `wtm.distributions`
   supports, mixtures included.  On a Beta prior the two agree by conjugacy.

Everything else -- the policy class, the search space, the backup formula, the
budget-exhaustion step, the utility accounting -- follows their code line for
line, and `algorithm1` is written to be read alongside it.

The guarantee being reproduced is theirs: Theorem 5.1 characterises the
optimal over-time allocation IN THE LIMIT OF MANY INDIVIDUALS, given the true
population distribution.  It is a mean-field oracle, not a finite-sample
learning procedure, and nothing here changes that.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import comb

import numpy as np

__all__ = ["CohortModel", "OverTimeResult", "algorithm1"]

SHIRALI_OFFSET = 0     # (1-p)^(2t - k):     t signals, t survivals
WTM_OFFSET = -1        # (1-p)^(2s - k - 1): s signals, s-1 survivals


@dataclass
class CohortModel:
    """Posterior expectations for cohort (k, t), from exact mixed moments.

    The cohort kernel is C(t,k) p^k (1-p)^(2t - k + offset).  Because every
    member of a cohort shares a posterior, a cohort's count may be multiplied
    by a posterior expectation -- which is what the backup formula does.
    """

    dist: object
    T: int
    offset: int = WTM_OFFSET

    def __post_init__(self):
        T = self.T
        shape = (T + 2, T + 2)
        self.surv2 = np.zeros(shape)   # E[(1-p)^2 | k, t]
        self.sig = np.zeros(shape)     # E[p(1-p)  | k, t]
        self.value = np.zeros(shape)   # 1 - E[(1-p)^(T-t) | k, t]
        self.mass = np.zeros(shape)    # unconditional cohort mass at t
        for t in range(1, T + 1):
            for k in range(0, t + 1):
                j = 2 * t - k + self.offset
                if j < 0:
                    continue
                base = self.dist.mixed(k, j)
                self.mass[k, t] = comb(t, k) * base
                if base <= 0:
                    continue
                self.surv2[k, t] = self.dist.mixed(k, j + 2) / base
                self.sig[k, t] = self.dist.mixed(k + 1, j + 1) / base
                self.value[k, t] = 1.0 - self.dist.mixed(k, j + self.T - t) / base


@dataclass
class OverTimeResult:
    utility: float
    q: np.ndarray
    t_hat: int
    treated: np.ndarray
    feasible: bool


def _simulate_traj(model: CohortModel, t0: int, init: np.ndarray,
                   q: np.ndarray) -> np.ndarray:
    """Algorithm 2.  Returns mass treated at each t, given thresholds q.

    Mirrors simulate_traj.m: at each step the untreated remainder of each
    cohort propagates by one survival and one signal draw, and the cohort at
    the threshold is treated.  The cohort index is vectorised -- the scalar
    form is kept in the git history and both were checked to agree to 1e-15.
    """
    T = model.T
    size = T + 2
    idx = np.arange(size)
    cur = np.zeros(size)
    cur[: len(init)] = init
    treated = np.zeros(T + 1)
    treated[t0] = cur[q[t0]] if q[t0] < size else 0.0

    for t in range(t0 + 1, T + 1):
        below = np.where(idx < q[t - 1], cur, 0.0)
        nxt = below * model.surv2[:size, t - 1]
        nxt[1:] += below[:-1] * model.sig[: size - 1, t - 1]
        cur = nxt
        treated[t] = cur[q[t]] if q[t] < size else 0.0
    return treated


def _valid_sequences(T: int):
    """Lemma 5.2: t_hat, q(1) in {0,1,2}, and a binary sequence of length T-1."""
    for t_hat in range(1, T):
        for k1 in range(0, 3):
            for code in range(0, 2 ** (T - 1)):
                bits = [(code >> i) & 1 for i in range(T - 1)]
                bits[t_hat - 1] += 1          # the extra step at t_hat
                q = np.cumsum([k1] + bits)
                yield t_hat, q


def algorithm1(dist, T: int, budget: float, offset: int = WTM_OFFSET,
               model: CohortModel = None) -> OverTimeResult:
    """Optimal over-time allocation, following Algorithm 1.

    `budget` is a population fraction (their B/N).  Returns the best feasible
    policy found over the search space of Lemma 5.2.
    """
    model = model or CohortModel(dist, T, offset)
    best = OverTimeResult(0.0, None, -1, None, False)

    for t_hat, q_cs in _valid_sequences(T):
        q = np.concatenate(([0], q_cs))       # 1-indexed by time

        init = np.zeros(T + 2)
        init[0] = model.mass[0, 1]
        init[1] = model.mass[1, 1]
        treated1 = _simulate_traj(model, 1, init, q)
        e_max = (model.mass[1, 1] if q[1] == 0 else 0.0) + treated1[1:T + 1].sum()

        # the same trajectory with the t_hat cohort left untreated
        init2 = np.zeros(T + 2)
        init2[q[t_hat]] = treated1[t_hat]
        q2 = q.copy()
        q2[t_hat] += 1
        treated2 = _simulate_traj(model, t_hat, init2, q2)
        d_e = treated1[t_hat] - treated2[1:T + 1].sum()

        if d_e == 0:
            continue
        rho = (e_max - budget) / d_e
        if rho > 1 or rho < 0:
            continue

        utility = 0.0
        treated = np.zeros(T + 1)
        for t in range(1, T + 1):
            if t == t_hat:
                treated[t] = (1 - rho) * treated1[t]
            else:
                treated[t] = treated1[t] + rho * treated2[t]
            utility += treated[t] * model.value[q[t], t] if q[t] <= T else 0.0

        if q[1] == 0:                          # the k=1 edge case at t=1
            treated[1] += model.mass[1, 1]
            utility += model.mass[1, 1] * model.value[1, 1]

        if utility > best.utility:
            best = OverTimeResult(float(utility), q[1:T + 1].copy(), t_hat,
                                  treated[1:T + 1].copy(), True)
    return best
