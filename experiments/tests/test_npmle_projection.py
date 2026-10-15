"""Regression tests for the NPMLE EM loop and the moment-cone projection.

The EM convergence test exists because of a real bug: `prev` was initialised to
-inf, so the check `ll - prev <= tol * max(1, abs(prev))` read `inf <= inf` and
returned "converged" after a single iteration.  The fit silently stopped at the
uniform starting weights and still reported `converged = True`.  These tests
would have caught it three ways: iteration count, monotone likelihood, and
moment recovery.

Run:  python experiments/tests/test_npmle_projection.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from wtm.distributions import BetaRisk
from wtm.moment_cone import cone_statistic, in_cone
from wtm.moment_projection import (empirical_path_counts, moments_with_covariance,
                                   path_indices, project_to_cone)
from wtm.npmle import fit_npmle, record_matrix
from wtm.observation import individual_functionals
from wtm.simulate import simulate_prefix
from wtm.moment_cone import build_path_alphabet, flatten_cohort_counts

GRID = np.linspace(0.0, 1.0, 200)
DIST = BetaRisk(1, 8)


def _counts(dist, t, N, seed):
    q = np.array([f.expectation(dist) for _, f in individual_functionals(t, 1)])
    return np.random.default_rng(seed).multinomial(N, q)


# -- NPMLE EM regression -----------------------------------------------------
def test_em_does_not_converge_on_first_iteration():
    """The original bug: `inf <= inf` reported convergence after one step."""
    for t in (2, 3, 4):
        n = _counts(DIST, t, 20000, 1)
        r = fit_npmle(n, GRID, t, 1)
        assert r.iterations > 1, (
            f"t={t}: EM stopped after {r.iterations} iteration(s) -- the "
            "first-pass convergence guard has regressed")


def test_em_likelihood_is_monotone():
    """EM must never decrease the log-likelihood."""
    for t in (2, 4):
        n = _counts(DIST, t, 20000, 2)
        Psi = record_matrix(GRID, t, 1)
        prev = -np.inf
        for iters in (1, 2, 5, 10, 50, 200):
            r = fit_npmle(n, GRID, t, 1, max_iter=iters, tol=0.0, Psi=Psi)
            assert r.loglik >= prev - 1e-9, (
                f"t={t}: log-likelihood fell from {prev} to {r.loglik} at "
                f"{iters} iterations")
            prev = r.loglik


def test_em_moves_away_from_the_uniform_start():
    """A converged fit must not still be the uniform initialisation."""
    for t in (2, 3, 4):
        n = _counts(DIST, t, 20000, 3)
        r = fit_npmle(n, GRID, t, 1)
        uniform = np.full(len(GRID), 1.0 / len(GRID))
        assert np.max(np.abs(r.weights - uniform)) > 1e-6, (
            f"t={t}: fitted weights are still uniform")


def test_em_recovers_moments():
    """At large N the fit must reproduce the low moments closely."""
    for t in (2, 3, 4):
        n = _counts(DIST, t, 200000, 4)
        r = fit_npmle(n, GRID, t, 1)
        for j in (1, 2):
            got = float(np.sum(r.weights * GRID ** j))
            want = DIST.raw_moment(j)
            assert abs(got - want) / want < 0.10, (
                f"t={t}: E[p^{j}] = {got:.6g}, expected {want:.6g}")


def test_em_weights_stay_on_the_simplex():
    for t in (2, 4):
        n = _counts(DIST, t, 20000, 5)
        r = fit_npmle(n, GRID, t, 1)
        assert np.all(r.weights >= -1e-15)
        assert abs(r.weights.sum() - 1.0) < 1e-10


def test_tighter_tolerance_runs_longer():
    n = _counts(DIST, 2, 20000, 6)
    loose = fit_npmle(n, GRID, 2, 1, tol=1e-4)
    tight = fit_npmle(n, GRID, 2, 1, tol=1e-12)
    assert tight.iterations >= loose.iterations, (
        "a tighter tolerance must not stop earlier than a looser one")


def test_convergence_flag_is_honest():
    """max_iter = 1 must not be reported as converged."""
    n = _counts(DIST, 2, 20000, 7)
    r = fit_npmle(n, GRID, 2, 1, max_iter=1, tol=1e-12)
    assert not r.converged, "one iteration reported as converged"


# -- path encoding -----------------------------------------------------------
def test_path_encoding_reproduces_cohort_counts():
    """Path counts must map back onto the observed cohort counts exactly."""
    for t in (2, 3, 4):
        data = simulate_prefix(DIST, 30000, 20, 1, t, seed=11)
        pc = empirical_path_counts(data, t)
        assert pc.sum() == data.N
        pa = build_path_alphabet(t)
        from_paths = pa.to_cohort.T @ pc
        direct = flatten_cohort_counts(data.cohort_counts(), t)
        assert np.max(np.abs(from_paths - direct)) == 0.0, f"t={t}"


# -- projection --------------------------------------------------------------
def test_projection_is_identity_inside_the_cone():
    data = simulate_prefix(DIST, 50000, 20, 1, 2, seed=12)
    m, cov = moments_with_covariance(empirical_path_counts(data, 2), 2)
    assert in_cone(m, 2), "t=2 estimate expected inside the cone at this N"
    r = project_to_cone(m, cov, np.linspace(0, 1, 201), 2)
    assert not r.was_projected
    assert np.array_equal(r.moments, m)


def test_projection_restores_feasibility():
    """The t>=3 estimates fall outside the cone; projection must repair them."""
    for t in (3, 4):
        data = simulate_prefix(DIST, 20000, 20, 1, t, seed=4242)
        m, cov = moments_with_covariance(empirical_path_counts(data, t), t)
        assert not in_cone(m, t), f"t={t}: expected the raw estimate outside"
        r = project_to_cone(m, cov, np.linspace(0, 1, 201), t)
        assert r.was_projected
        assert cone_statistic(r.moments, t) >= -1e-9, (
            f"t={t}: projected moments still outside the cone")


def test_projection_moves_the_high_moments_most():
    """Covariance weighting must spend its budget on the noisy high moments."""
    data = simulate_prefix(DIST, 20000, 20, 1, 4, seed=4242)
    m, cov = moments_with_covariance(empirical_path_counts(data, 4), 4)
    r = project_to_cone(m, cov, np.linspace(0, 1, 201), 4)
    assert r.shifted_moment_index >= 5, (
        f"largest shift landed on m_{r.shifted_moment_index}, expected a high one")


def test_covariance_weighting_beats_identity_on_low_moments():
    data = simulate_prefix(DIST, 20000, 20, 1, 4, seed=4242)
    m, cov = moments_with_covariance(empirical_path_counts(data, 4), 4)
    grid = np.linspace(0, 1, 201)
    true = np.array([DIST.raw_moment(j) for j in range(8)])

    def low_err(weighting):
        r = project_to_cone(m, cov, grid, 4, weighting=weighting)
        return np.max(np.abs(r.moments[1:4] - true[1:4]) / true[1:4])

    assert low_err("covariance") < low_err("identity"), (
        "covariance weighting should preserve well-estimated low moments better")


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
