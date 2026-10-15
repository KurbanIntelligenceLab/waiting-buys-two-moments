"""Tests for experiments/em_mechanism (batch EM trajectories)."""
import numpy as np
import pytest

from experiments.em_mechanism.core import (DELTA, INIT_NAMES, Welfare, certified_npmle,
                                           exact_cell_probs, grid_and_psi, init_weights,
                                           run_em, schedule)
from wtm.distributions import BetaRisk
from wtm.npmle import fit_npmle
from wtm.regret_calibrated import _trajectory


@pytest.fixture(scope="module")
def setup():
    grid, Psi = grid_and_psi(200, 2)
    rng = np.random.default_rng(3)
    counts = rng.multinomial(1000, [0.08, 0.02, 0.7, 0.17, 0.03], size=6).astype(float)
    return grid, Psi, counts


def test_schedule_contains_paper_counts():
    s = schedule(10_000)
    assert {0, 30, 55, 80, 100, 10_000} <= set(s.tolist())
    assert np.all(np.diff(s) > 0) and np.all(s[:101] == np.arange(101))


@pytest.mark.parametrize("name", INIT_NAMES)
def test_initializations_positive_on_simplex(name):
    grid, _ = grid_and_psi(200, 2)
    w = init_weights(name, grid)
    assert w.shape == (200,) and np.all(w > 0) and abs(w.sum() - 1) < 1e-12


def test_batch_em_reproduces_fit_npmle(setup):
    grid, Psi, counts = setup
    ev = Welfare(grid, 20, 0.002)
    W0 = np.tile(init_weights("uniform", grid), (len(counts), 1))
    run = run_em(counts, Psi, grid, W0, np.arange(0, 81), {"e": ev}, snapshot_at=(30, 55, 80))
    for K in (30, 55, 80):
        for b in range(len(counts)):
            f = fit_npmle(counts[b], grid, 2, 1, tol=1e-300, max_iter=K, Psi=Psi)
            assert f.iterations == K
            assert np.allclose(f.weights, run.snapshots[K][b], rtol=1e-12, atol=1e-15)
            d = int(np.argmax(_trajectory(f.weights, ev.cache, 0.002))) + 1
            assert d == run.dates["e"][K, b]
    assert run.mono_viol.max() == 0


def test_kkt_bounds_likelihood_gap(setup):
    grid, Psi, counts = setup
    W0 = np.tile(init_weights("uniform", grid), (len(counts), 1))
    run = run_em(counts, Psi, grid, W0, schedule(2000), {})
    for b in range(len(counts)):
        fit = certified_npmle(counts[b], grid, 2, Psi)
        best = max(fit.ll, run.ll[-1, b])
        # concavity: ll* - ll_k <= kkt_k  (up to the reference's own residual)
        assert np.all(best - run.ll[:, b] <= run.kkt[:, b] + max(fit.kkt, 0) + 1e-12)


def test_welfare_batch_matches_trajectory(setup):
    grid, _, _ = setup
    rng = np.random.default_rng(1)
    ev = Welfare(grid, 12, 0.01)
    W = rng.dirichlet(np.full(200, 0.5), size=10)
    C = ev.curves(W)
    for b in range(10):
        assert np.allclose(C[b], _trajectory(W[b], ev.cache, 0.01), rtol=1e-13, atol=0)


def test_exact_cell_probs_form_a_distribution():
    for tau in (2, 3, 4):
        q = exact_cell_probs(BetaRisk(a=1.0, b=8.0, name="b"), tau)
        assert np.all(q > 0) and abs(q.sum() - 1) < 1e-12
    assert DELTA == 1e-8


def test_reported_claims_verify_against_saved_outputs():
    """Every number in REPORT.md, MECHANISM_REPORT.md and suggested_latex.tex is
    recomputed from saved outputs and checked.  Missing outputs FAIL this test
    (verify_claims exits 2 naming them): in a release, the EM data archive must
    be extracted alongside the main archive."""
    import subprocess
    import sys
    r = subprocess.run([sys.executable, "-m", "experiments.em_mechanism.verify_claims"],
                       capture_output=True, text=True)
    assert r.returncode == 0, (r.stdout + r.stderr)[-3000:]
