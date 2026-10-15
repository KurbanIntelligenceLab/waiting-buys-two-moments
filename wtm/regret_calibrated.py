"""The regret-calibrated randomized rule (frozen spec: REGRET_RULE_SPEC.md).

Candidate A with randomized stopping.  NPMLE trajectory -> relative adjacent
contrast -> paired multinomial bootstrap standard error -> studentized statistic
-> probit wait probability -> forward randomized stopping walk.

Every constant here is the frozen one; none was chosen by looking at benchmark
performance, and the benchmark has not been run.  `lambda_ = 1.0` is the
posterior anchor (at lambda = 1, pi_s is the flat-prior posterior probability
that waiting one more period helps).  `ensemble_size = 200` follows from the
relative error of a standard-deviation estimate, 1/sqrt(2(E-1)) = 5%.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.stats import norm

from .lp import group_matrices
from .npmle import fit_npmle, record_matrix
from .welfare import fill_greedy

LAMBDA_DEFAULT = 1.0          # frozen: posterior anchor
ENSEMBLE_DEFAULT = 200        # frozen: 5% relative error on the SD estimate
EPS_SE = 1e-6                 # relative-contrast units
NPMLE_GRID = 200
NPMLE_TOL = 1e-8
NPMLE_MAX_ITER = 20000        # raised from 5000: 10% of gamma=1 fits hit the
                              # old cap, which would have shown up in Table 4
                              # as a spurious no-estimate rate. Worst observed
                              # requirement is 7,522 iterations. Raising a cap
                              # can only turn non-convergence into convergence;
                              # it never changes an already-converged answer.
MIN_BOOTSTRAP_FRACTION = 0.8  # below this, the run is a no-estimate


@dataclass
class RegretRuleOutput:
    selected_date: int
    success: bool = True
    failure_reason: str = ""
    wait_probabilities: np.ndarray = field(repr=False, default=None)
    z: np.ndarray = field(repr=False, default=None)
    contrasts: np.ndarray = field(repr=False, default=None)
    standard_errors: np.ndarray = field(repr=False, default=None)
    n_bootstrap_used: int = 0
    n_bootstrap_failed: int = 0
    npmle_iterations: int = 0
    npmle_converged: bool = False
    randomized: bool = True


def _trajectory(w, cache, beta):
    """W^s for every candidate date, from grid weights."""
    return np.array([fill_greedy(PHI @ w, UTIL @ w, beta)[0]
                     for PHI, UTIL in cache])


def _contrasts(W):
    """Relative adjacent contrasts (W^{s+1} - W^s) / W^max."""
    m = W.max()
    if not np.isfinite(m) or m <= 0:
        return None
    return np.diff(W) / m


def forward_regret(W):
    """Global forward regret, variant 2 (REGRET_RULE_V2_SPEC.md, frozen a12d6c0).

        R_s = (max_{u>s} W^u - W^s) / max_u W^u,   s = 1 .. T-2

    The welfare still available by continuing past s, in the same units as the
    adjacent contrast `_contrasts`, to which it reduces when the forward maximum
    is attained at the next date (the unimodal case).  R_s >= D_s always.
    """
    m = W.max()
    if not np.isfinite(m) or m <= 0:
        return None
    W = np.asarray(W, dtype=float)
    # suffix maxima: fwd[s] = max over u > s, for s = 0 .. len(W)-2
    fwd = np.maximum.accumulate(W[::-1])[::-1][1:]
    return (fwd - W[:-1]) / m


def regret_calibrated(record_counts, T: int, gamma: int, t: int, beta: float,
                      rule_seed: int, lambda_: float = LAMBDA_DEFAULT,
                      ensemble_size: int = ENSEMBLE_DEFAULT,
                      grid_size: int = NPMLE_GRID, tol: float = NPMLE_TOL,
                      max_iter: int = NPMLE_MAX_ITER,
                      link: str = "probit",
                      statistic: str = "adjacent") -> RegretRuleOutput:
    """See REGRET_RULE_SPEC.md.  `record_counts` are single-record atom counts."""
    grid = np.linspace(0.0, 1.0, grid_size)
    Psi = record_matrix(grid, t, gamma)
    cache = [group_matrices(grid, int(s), gamma, T) for s in range(1, T)]
    n = np.asarray(record_counts, dtype=float)
    N = int(round(n.sum()))

    # S1-S3: fit, trajectory, contrasts
    fit = fit_npmle(n, grid, t, gamma, tol=tol, max_iter=max_iter, Psi=Psi)
    if not fit.converged:
        return RegretRuleOutput(-1, False, "npmle_no_convergence",
                                npmle_iterations=fit.iterations)
    W = _trajectory(fit.weights, cache, beta)
    _stat = _contrasts if statistic == "adjacent" else forward_regret
    D = _stat(W)
    if D is None:
        return RegretRuleOutput(-1, False, "degenerate_trajectory",
                                npmle_iterations=fit.iterations,
                                npmle_converged=True)

    # S4: paired multinomial bootstrap, differences taken replicate-wise
    rng = np.random.default_rng(int(rule_seed))
    q_hat = n / N
    boot, n_fail = [], 0
    for _ in range(int(ensemble_size)):
        nb = rng.multinomial(N, q_hat)
        fb = fit_npmle(nb, grid, t, gamma, tol=tol, max_iter=max_iter,
                       Psi=Psi, w0=fit.weights)
        if not fb.converged:
            n_fail += 1
            continue
        Db = _stat(_trajectory(fb.weights, cache, beta))
        if Db is not None:
            boot.append(Db)
    if len(boot) < MIN_BOOTSTRAP_FRACTION * ensemble_size:
        return RegretRuleOutput(-1, False, "bootstrap_failure_rate_above_20pct",
                                n_bootstrap_used=len(boot),
                                n_bootstrap_failed=n_fail,
                                npmle_iterations=fit.iterations,
                                npmle_converged=True)
    se = np.std(np.vstack(boot), axis=0, ddof=1)

    # S5: studentize.  0/0 is defined as 0 (maximal hedge), never nan.
    z = np.where((se <= 0) & (D == 0), 0.0, D / np.maximum(se, EPS_SE))

    # S6-S7: smooth wait probability, then the randomized stopping walk
    pi = norm.cdf(z / lambda_) if link == "probit" else 1.0 / (1.0 + np.exp(-z / lambda_))
    u = rng.random(len(pi))
    selected = int(T - 1)
    for i, s in enumerate(range(1, T - 1)):
        if u[i] > pi[i]:
            selected = s
            break

    return RegretRuleOutput(selected, True, "", wait_probabilities=pi, z=z,
                            contrasts=D, standard_errors=se,
                            n_bootstrap_used=len(boot), n_bootstrap_failed=n_fail,
                            npmle_iterations=fit.iterations, npmle_converged=True)


def record_counts_from_prefix(data, t: int) -> np.ndarray:
    """Single-record atom counts from a simulated prefix -- the length-t window.

    Atom order matches `observation.individual_functionals(t, gamma)`:
    (exit at end of s, y^s = k) for s = 1..t-1, then (active in period t, y^t = k).

    The period-t exit is deliberately not consulted.  `active_mask[t-1]` is the
    pool active at the *start* of period t -- everyone who survived the exits at
    the ends of periods 1..t-1 -- which is what a length-t prefix reveals.  Using
    `exited_by == -1` instead would condition on surviving the period-t exit too,
    which is the Window-B model the R15 correction removed: it adds a factor of
    (1-p), pushing the kernel to degree 2t and identifying one moment beyond
    Pi_{2t-1}.  See results/r15_window_correction_report.md.
    """
    counts = []
    for s in range(1, t):
        for k in range(s + 1):
            counts.append(int(np.sum((data.exited_by == s)
                                     & (data.y_by_period[s - 1] == k))))
    for k in range(t + 1):
        counts.append(int(np.sum(data.active_mask[t - 1]
                                 & (data.y_by_period[t - 1] == k))))
    return np.array(counts, dtype=float)


def expected_regret_rule(boot_trajectories):
    """Variant 3: global expected-regret minimisation (REGRET_RULE_V3_SPEC.md).

        rho_b(s) = (max_u W_b^u - W_b^s) / max_u W_b^u
        t_hat    = argmin_s  mean_b rho_b(s),  ties -> earliest date

    `boot_trajectories` is a (B, T-1) array of bootstrap welfare trajectories.
    Deterministic: no walk, no randomisation, no tuning constant.  Its
    zero-uncertainty limit is exactly the NPMLE argmax (spec section 3).
    """
    Wb = np.asarray(boot_trajectories, dtype=float)
    if Wb.ndim != 2 or Wb.shape[0] == 0:
        return -1, None
    m = Wb.max(axis=1, keepdims=True)
    if not np.isfinite(m).all() or (m <= 0).any():
        return -1, None
    rho = (m - Wb) / m
    rbar = rho.mean(axis=0)
    return int(np.argmin(rbar)) + 1, rbar
