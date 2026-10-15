"""Validation suite (brief Section 4).  Must pass before any experiment runs.

Run with:  python -m pytest experiments/tests -q
      or:  python experiments/tests/test_validation.py
"""

from __future__ import annotations

import sys
from fractions import Fraction
from pathlib import Path

import numpy as np
from scipy import integrate

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from wtm.distributions import BetaRisk, DiscreteRisk, MixtureRisk
from wtm.lp import IdentifiedSet, support_grid, truth_seeds
from wtm.observation import phi, phi_exit, phi_survivor, ptilde, utility_poly
from wtm.rules import RuleContext, act_immediately, evaluate, oracle
from wtm.simulate import simulate_prefix
from wtm.welfare import (PopulationConfig, group_masses, group_utilities,
                         welfare_full_info, welfare_y)

BETAS = [BetaRisk(1, 32), BetaRisk(1, 8), BetaRisk(1, 4)]
GAMMAS = (1, 2, 4)


# -- 4.1 probability distributions -------------------------------------------
def test_support_and_normalisation():
    rng = np.random.default_rng(0)
    for d in BETAS:
        x = d.sample(rng, 20000)
        assert np.all((x >= 0.0) & (x <= 1.0)), f"{d.name} sampled outside [0,1]"
        assert abs(d.mixed(0, 0) - 1.0) < 1e-12, f"{d.name} does not integrate to 1"
        mass, _ = integrate.quad(lambda p: d.pdf(p), 0.0, 1.0, limit=200)
        assert abs(mass - 1.0) < 1e-9

    disc = DiscreteRisk(points=(0.01, 0.5, 0.9), weights=(0.5, 0.3, 0.2))
    assert abs(disc.mixed(0, 0) - 1.0) < 1e-12
    mix = MixtureRisk(components=(BetaRisk(1, 32), BetaRisk(5, 5)), weights=(0.9, 0.1))
    assert abs(mix.mixed(0, 0) - 1.0) < 1e-12
    assert abs(mix.raw_moment(1) - (0.9 / 33 + 0.1 * 0.5)) < 1e-12


# -- 4.2 Beta moments --------------------------------------------------------
def test_beta_moments_match_analytic():
    for d in BETAS:
        a, b = d.a, d.b
        for j in range(1, 9):
            # E[p^j] = prod_{r<j} (a+r)/(a+b+r)
            analytic = np.prod([(a + r) / (a + b + r) for r in range(j)])
            assert abs(d.raw_moment(j) - analytic) < 1e-13 * max(1.0, analytic)
        for i in range(4):
            for j in range(6):
                exact = float(d.mixed_exact(i, j))
                assert abs(d.mixed(i, j) - exact) <= 1e-12 * max(1e-300, abs(exact))
        # E[(1-p)^m] = b/(b+m) for Beta(1,b)
        for m in range(1, 12):
            assert d.q_exact(m) == Fraction(int(b), int(b) + m)


def test_moment_polynomials_are_cancellation_free():
    """Every phi and u has nonnegative coefficients in the mixed basis."""
    for gamma in GAMMAS:
        for s in range(1, 7):
            for k in range(s + 1):
                assert phi(s, k, gamma).all_nonnegative
                assert phi_exit(s, k, gamma).all_nonnegative
                assert phi_survivor(s, k, gamma).all_nonnegative
    for s in range(1, 20):
        assert utility_poly(s, 20).all_nonnegative


def test_phi_partitions_the_active_pool():
    """sum_k phi_{s,k}(p) = (1-p)^(s-1) exactly, at every gamma."""
    p = np.linspace(0.0, 1.0, 101)
    for gamma in GAMMAS:
        for s in range(1, 8):
            total = sum(phi(s, k, gamma).evaluate(p) for k in range(s + 1))
            assert np.max(np.abs(total - (1 - p) ** (s - 1))) < 1e-12


# -- 4.3 simulation vs mean field --------------------------------------------
def test_simulation_matches_mean_field():
    N, t, T = 400_000, 4, 20
    for d in (BetaRisk(1, 8),):
        for gamma in GAMMAS:
            data = simulate_prefix(d, N, T, gamma, t, seed=12345)

            # survival fraction: P[active in period s] = E[(1-p)^(s-1)]
            emp = data.empirical_active_fraction()
            for s in range(1, t + 1):
                theory = d.mixed(0, s - 1)
                se = np.sqrt(max(theory * (1 - theory), 1e-12) / N)
                assert abs(emp[s - 1] - theory) < 5 * se, f"survival s={s}"

            # observation-count distribution: C[s,k]/N -> E[phi_{s,k}(p)]
            C = data.cohort_counts() / N
            for s in range(1, t + 1):
                m = group_masses(d, s, gamma)
                for k in range(s + 1):
                    se = np.sqrt(max(m[k] * (1 - m[k]), 1e-12) / N)
                    assert abs(C[s - 1, k] - m[k]) < 5 * se + 1e-9, f"phi s={s} k={k}"

            # exit distribution: P[exit at end of s with y^s = k]
            for s in range(1, t + 1):
                for k in range(s + 1):
                    got = np.mean((data.exited_by == s)
                                  & (data.y_by_period[s - 1] == k)
                                  & data.active_mask[s - 1])
                    theory = phi_exit(s, k, gamma).expectation(d)
                    se = np.sqrt(max(theory * (1 - theory), 1e-12) / N)
                    assert abs(got - theory) < 5 * se + 1e-9, f"exit s={s} k={k}"

            # risk moments of the drawn population
            for j in (1, 2, 3):
                assert abs(np.mean(data.p_true ** j) - d.raw_moment(j)) < 0.02


def test_ptilde_definition():
    p = np.linspace(0, 1, 11)
    assert np.allclose(ptilde(p, 1), p)
    assert np.allclose(ptilde(p, 2), 1 - (1 - p) ** 2)


# -- 4.4 welfare -------------------------------------------------------------
def test_welfare_below_full_information():
    for d in BETAS:
        for gamma in GAMMAS:
            for T in (12, 20):
                for beta in (0.0005, 0.002, 0.01):
                    for s in range(1, T):
                        w = welfare_y(d, s, gamma, T, beta)
                        wb = welfare_full_info(d, s, T, beta)
                        assert w <= wb + 1e-15, f"W>Wbar at {d.name} s={s}"
                        assert w >= 0.0 and wb <= beta + 1e-15


def test_full_information_welfare_is_decreasing():
    for d in BETAS:
        for T in (12, 20):
            for beta in (0.0005, 0.002, 0.01):
                wb = np.array([welfare_full_info(d, s, T, beta) for s in range(1, T)])
                assert np.all(np.diff(wb) <= 1e-15), f"Wbar not decreasing ({d.name})"


def test_greedy_allocation_respects_budget():
    from wtm.welfare import fill_greedy
    for d in BETAS:
        for gamma in GAMMAS:
            for s in (1, 3, 7):
                m = group_masses(d, s, gamma)
                v = group_utilities(d, s, gamma, 20)
                _, a = fill_greedy(m, v, 0.002)
                assert np.all((a >= -1e-15) & (a <= 1 + 1e-15))
                assert np.dot(a, m) <= 0.002 + 1e-15


# -- 4.5 / 4.6 reference rules ----------------------------------------------
def test_oracle_is_perfect_and_act_immediately_is_date_one():
    effs, regrets, failures = [], [], 0
    for d in BETAS:
        for gamma in GAMMAS:
            for beta in (0.0005, 0.002, 0.01):
                pop = PopulationConfig(d, gamma, 20, beta)
                data = simulate_prefix(d, 2000, 20, gamma, 2, seed=1)
                ctx = RuleContext(T=20, beta=beta, gamma=gamma, t=2, truth=pop)

                out = oracle(data, ctx)
                sc = evaluate(out.selected_date, pop)
                effs.append(sc["welfare_efficiency"])
                regrets.append(sc["relative_regret"])
                failures += 0 if out.success else 1
                assert sc["exact_argmax_match"]

                imm = act_immediately(data, ctx)
                assert imm.selected_date == 1

    effs = np.array(effs)
    assert float(np.mean(np.array(regrets) ** 2)) == 0.0        # MSR
    assert float(np.median(effs)) == 1.0
    assert float(np.quantile(effs, 0.10)) == 1.0
    assert float(np.min(effs)) == 1.0
    assert failures == 0


# -- 4.7 paired inputs -------------------------------------------------------
def test_all_rules_receive_identical_data():
    from wtm.rules import AVAILABLE, REGISTRY
    d, gamma, T, beta, t = BetaRisk(1, 8), 1, 20, 0.002, 3
    pop = PopulationConfig(d, gamma, T, beta)
    data = simulate_prefix(d, 5000, T, gamma, t, seed=99)
    ctx = RuleContext(T=T, beta=beta, gamma=gamma, t=t, truth=pop)

    fingerprint = (data.y.tobytes(), data.exited_by.tobytes(),
                   data.active_mask.tobytes())
    for name in AVAILABLE:
        REGISTRY[name](data, ctx)
        assert (data.y.tobytes(), data.exited_by.tobytes(),
                data.active_mask.tobytes()) == fingerprint, f"{name} mutated the data"
    assert len(AVAILABLE) >= 2


# -- 4.8 numerical reproducibility -------------------------------------------
def test_same_seed_reproduces_bitwise():
    d = BetaRisk(1, 4)
    for gamma in GAMMAS:
        a = simulate_prefix(d, 20000, 20, gamma, 4, seed=777)
        b = simulate_prefix(d, 20000, 20, gamma, 4, seed=777)
        assert np.array_equal(a.y, b.y)
        assert np.array_equal(a.exited_by, b.exited_by)
        assert np.array_equal(a.active_mask, b.active_mask)
        assert np.array_equal(a.p_true, b.p_true)
        c = simulate_prefix(d, 20000, 20, gamma, 4, seed=778)
        assert not np.array_equal(a.y, c.y)


def test_mean_field_trajectory_is_deterministic():
    d = BetaRisk(1, 8)
    p1 = PopulationConfig(d, 1, 20, 0.002)
    p2 = PopulationConfig(BetaRisk(1, 8), 1, 20, 0.002)
    assert np.array_equal(p1.W, p2.W) and p1.t_star == p2.t_star


# -- LP formulation ----------------------------------------------------------
def test_lp_bounds_contain_the_truth():
    d, gamma, T, beta = BetaRisk(1, 8), 1, 20, 0.002
    pop = PopulationConfig(d, gamma, T, beta)
    S = IdentifiedSet(d, 2, support_grid(201))
    w, _ = S.feasible_point()
    assert w is not None and S.moment_residual(w) < 1e-9
    for s in (2, 5, 9):
        profile, lam = truth_seeds(d, s, gamma, T, beta)
        r = S.bounds(s, gamma, T, beta, truth_profile=profile, truth_lambda=lam,
                     n_fill=9, n_lambda=15, n_refine=5)
        true_W = pop.welfare_at(s)
        assert r.lower <= true_W + 1e-12, f"lower bound above truth at s={s}"
        assert true_W <= r.upper + 1e-12, f"upper bound below truth at s={s}"
        assert r.lower <= r.upper


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
