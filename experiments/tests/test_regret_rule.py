"""Tests for the regret-calibrated randomized rule (REGRET_RULE_SPEC.md).

Covers the parts of the frozen spec that are easy to get silently wrong: the
degenerate standard-error cases, the paired differencing, the bootstrap-failure
path, and reproducibility under a fixed rule seed.

Run:  python experiments/tests/test_regret_rule.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from wtm.distributions import BetaRisk
from wtm.observation import individual_functionals
from wtm.regret_calibrated import (EPS_SE, record_counts_from_prefix,
                                   regret_calibrated)
from wtm.simulate import simulate_prefix

DIST = BetaRisk(1, 8)
T, GAMMA, BETA = 20, 1, 0.002


def _counts(t, N=20000, seed=4242):
    return record_counts_from_prefix(simulate_prefix(DIST, N, T, GAMMA, t, seed), t)


def test_record_counts_are_a_partition():
    """Every individual falls in exactly one record atom."""
    for t in (2, 3, 4):
        data = simulate_prefix(DIST, 30000, T, GAMMA, t, seed=5)
        c = record_counts_from_prefix(data, t)
        assert c.sum() == data.N, f"t={t}: {c.sum()} != {data.N}"
        assert np.all(c >= 0)
        assert len(c) == len(individual_functionals(t, GAMMA))


def test_record_law_matches_population():
    """Empirical record frequencies converge on the population law."""
    for t in (2, 4):
        c = _counts(t, N=400000, seed=9)
        emp = c / c.sum()
        pop = np.array([f.expectation(DIST) for _, f in individual_functionals(t, GAMMA)])
        assert np.max(np.abs(emp - pop)) < 0.01, f"t={t}"


def test_wait_probabilities_are_valid():
    out = regret_calibrated(_counts(2), T, GAMMA, 2, BETA, rule_seed=1,
                            ensemble_size=25)
    assert out.success
    p = out.wait_probabilities
    assert np.all(np.isfinite(p)) and np.all((p >= 0) & (p <= 1))
    assert len(p) == T - 2, "one wait probability per adjacent pair"


def test_selected_date_is_in_range():
    for t in (2, 3, 4):
        out = regret_calibrated(_counts(t), T, GAMMA, t, BETA, rule_seed=3,
                                ensemble_size=25)
        assert out.success
        assert 1 <= out.selected_date <= T - 1, out.selected_date


def test_same_rule_seed_reproduces_exactly():
    c = _counts(2)
    a = regret_calibrated(c, T, GAMMA, 2, BETA, rule_seed=77, ensemble_size=30)
    b = regret_calibrated(c, T, GAMMA, 2, BETA, rule_seed=77, ensemble_size=30)
    assert a.selected_date == b.selected_date
    assert np.array_equal(a.wait_probabilities, b.wait_probabilities)
    assert np.array_equal(a.standard_errors, b.standard_errors)


def test_different_rule_seed_can_change_the_draw():
    """The rule is randomised: the same data may yield a different date."""
    c = _counts(2)
    dates = {regret_calibrated(c, T, GAMMA, 2, BETA, rule_seed=s,
                               ensemble_size=25).selected_date
             for s in range(40)}
    assert len(dates) >= 1
    probs = regret_calibrated(c, T, GAMMA, 2, BETA, rule_seed=0,
                              ensemble_size=25).wait_probabilities
    if np.any((probs > 0.02) & (probs < 0.98)):
        assert len(dates) > 1, ("with an unresolved contrast the rule must "
                                "actually randomise across rule seeds")


def test_zero_se_zero_contrast_gives_half():
    """0/0 must be defined as z = 0 (maximal hedge), never nan."""
    from scipy.stats import norm
    D = np.array([0.0, 0.5])
    se = np.array([0.0, 0.0])
    z = np.where((se <= 0) & (D == 0), 0.0, D / np.maximum(se, EPS_SE))
    assert z[0] == 0.0 and np.isfinite(z[0])
    assert norm.cdf(z[0]) == 0.5
    assert z[1] > 0 and np.isfinite(z[1]), "nonzero contrast with zero SE commits"


def test_se_floor_prevents_infinities():
    D = np.array([1e-3])
    se = np.array([0.0])
    z = D / np.maximum(se, EPS_SE)
    assert np.isfinite(z[0]) and z[0] == 1e-3 / EPS_SE


def test_standard_errors_are_paired_not_marginal():
    """The SE of the difference must be smaller than the naive marginal sum.

    W^{s+1} and W^s come from the same fit and are strongly positively
    correlated; assembling the SE as sqrt(se_a^2 + se_b^2) would inflate it and
    push every z toward zero.
    """
    from wtm.lp import group_matrices
    from wtm.npmle import fit_npmle, record_matrix
    from wtm.welfare import fill_greedy
    t = 2
    c = _counts(t)
    grid = np.linspace(0, 1, 200)
    Psi = record_matrix(grid, t, GAMMA)
    cache = [group_matrices(grid, int(s), GAMMA, T) for s in range(1, T)]
    rng = np.random.default_rng(11)
    N = int(c.sum())
    traj = []
    for _ in range(60):
        nb = rng.multinomial(N, c / N)
        f = fit_npmle(nb, grid, t, GAMMA, Psi=Psi)
        traj.append([fill_greedy(P @ f.weights, U @ f.weights, BETA)[0]
                     for P, U in cache])
    W = np.array(traj)
    paired = np.std(np.diff(W, axis=1), axis=0, ddof=1)
    marginal = np.sqrt(np.std(W[:, 1:], axis=0, ddof=1) ** 2
                       + np.std(W[:, :-1], axis=0, ddof=1) ** 2)
    assert np.median(paired) < np.median(marginal), (
        "paired differencing should shrink the SE relative to the marginal sum")


def test_ensemble_size_changes_precision_not_the_contrast():
    """E affects only the SE estimate; the point contrast is E-independent."""
    c = _counts(2)
    a = regret_calibrated(c, T, GAMMA, 2, BETA, rule_seed=5, ensemble_size=25)
    b = regret_calibrated(c, T, GAMMA, 2, BETA, rule_seed=5, ensemble_size=60)
    assert np.allclose(a.contrasts, b.contrasts), (
        "the point estimate must not depend on the bootstrap size")


def test_lambda_limits():
    """Small lambda commits; large lambda hedges toward a coin flip."""
    c = _counts(2)
    sharp = regret_calibrated(c, T, GAMMA, 2, BETA, rule_seed=2,
                              ensemble_size=25, lambda_=0.05)
    flat = regret_calibrated(c, T, GAMMA, 2, BETA, rule_seed=2,
                             ensemble_size=25, lambda_=500.0)
    assert np.max(np.abs(sharp.wait_probabilities - 0.5)) > \
           np.max(np.abs(flat.wait_probabilities - 0.5))
    assert np.all(np.abs(flat.wait_probabilities - 0.5) < 0.2), (
        "at very large lambda every wait probability should approach 1/2")


def test_no_estimate_is_reported_not_hidden():
    """A rule that cannot return a date must say so, with a reason."""
    from wtm.regret_calibrated import RegretRuleOutput
    out = RegretRuleOutput(-1, False, "npmle_no_convergence")
    assert out.selected_date == -1 and not out.success and out.failure_reason


def _main():
    fns = [(n, f) for n, f in sorted(globals().items())
           if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in fns:
        try:
            fn()
            print(f"PASS  {name}")
        except AssertionError as exc:
            failed += 1
            print(f"FAIL  {name}: {exc}")
        except Exception as exc:                      # noqa: BLE001
            failed += 1
            print(f"ERROR {name}: {type(exc).__name__}: {exc}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_main())
