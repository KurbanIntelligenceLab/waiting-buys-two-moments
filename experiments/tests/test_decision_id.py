"""Tests for the decision-identification reduction and its certificates.

Each test checks one step of results/decision_identification/REDUCTION.md, and
the docstrings name the lemma.  The certificate tests are adversarial: they
check that an exact certificate can never undercut a value that some member
of M actually attains, including when the multipliers are deliberately
corrupted.

Run:  python experiments/tests/test_decision_id.py
"""

from __future__ import annotations

import random
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np
from scipy.optimize import linprog

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from wtm.distributions import BetaRisk, DiscreteRisk, MixtureRisk  # noqa: E402
from wtm.decision_id import (DecisionProblem, bernstein_lower_bound,  # noqa: E402
                             bound_pair, build_exact_witness, exact_greedy,
                             float_welfare, mono_to_bernstein, nested_alloc,
                             solve_box_lp, verify_box_certificate, GridState,
                             eval_mixed_exact)
from wtm.observation import phi, utility_poly  # noqa: E402
from wtm.welfare import welfare_y  # noqa: E402

RNG = np.random.default_rng(20260925)
random.seed(20260925)


def _random_rational_measure(n_atoms):
    xs = [Fraction(random.randint(0, 1000), 1000) for _ in range(n_atoms)]
    ws = [Fraction(random.randint(0, 50)) for _ in range(n_atoms)]
    if random.random() < 0.3:
        ws[0] = Fraction(0)
    tot = sum(ws) or Fraction(1)
    return xs, [w / tot for w in ws] if sum(ws) else [Fraction(1, n_atoms)] * n_atoms


def _groups_exact(xs, ws, s, T):
    u = utility_poly(s, T)
    m = [sum((w * eval_mixed_exact(phi(s, k), x) for x, w in zip(xs, ws)), Fraction(0))
         for k in range(s + 1)]
    v = [sum((w * eval_mixed_exact(phi(s, k) * u, x) for x, w in zip(xs, ws)), Fraction(0))
         for k in range(s + 1)]
    return m, v


def test_group_ratio_monotone():
    """Lemma 1, step 1: rho_k is nondecreasing in k for every Q (exact)."""
    for _ in range(60):
        xs, ws = _random_rational_measure(random.randint(1, 6))
        T = random.choice([12, 20]); s = random.randint(1, T - 1)
        m, v = _groups_exact(xs, ws, s, T)
        pos = [k for k in range(s + 1) if m[k] > 0]
        for a, b in zip(pos, pos[1:]):
            assert v[b] * m[a] >= v[a] * m[b]
        assert all(0 <= vk <= mk for vk, mk in zip(v, m))


def test_threshold_matches_greedy():
    """Lemma 1: max over nested threshold allocations == greedy welfare (exact)."""
    for _ in range(60):
        xs, ws = _random_rational_measure(random.randint(1, 6))
        T = random.choice([12, 20]); s = random.randint(1, T - 1)
        beta = random.choice([Fraction(1, 2000), Fraction(1, 100), Fraction(1, 2),
                              Fraction(3, 1)])                 # includes beta > mass
        m, v = _groups_exact(xs, ws, s, T)
        Wg = exact_greedy(m, v, beta)
        # threshold family: the best feasible theta; candidates are integer
        # breakpoints and the exact budget-binding fill in each piece
        best = Fraction(0)
        for j in range(s + 1):
            for f in (Fraction(0), Fraction(1)):
                a = nested_alloc(Fraction(j) + f, s)
                if sum(ak * mk for ak, mk in zip(a, m)) <= beta:
                    best = max(best, sum(ak * vk for ak, vk in zip(a, v)))
            a0 = nested_alloc(Fraction(j), s)
            k = s - j
            used = sum(ak * mk for ak, mk in zip(a0, m))
            if m[k] > 0 and used <= beta:
                f = min(Fraction(1), (beta - used) / m[k])
                a = nested_alloc(Fraction(j) + f, s)
                best = max(best, sum(ak * vk for ak, vk in zip(a, v)))
        assert best == Wg, (best, Wg)
        # greedy is also >= every random feasible allocation
        for _ in range(20):
            a = [Fraction(random.randint(0, 10), 10) for _ in range(s + 1)]
            if sum(ak * mk for ak, mk in zip(a, m)) <= beta:
                assert sum(ak * vk for ak, vk in zip(a, v)) <= Wg


def test_dual_form():
    """Lemma 2: W = min_{lambda in [0,1]} f(lambda); the minimiser is a group ratio."""
    for _ in range(60):
        xs, ws = _random_rational_measure(random.randint(1, 6))
        T = random.choice([12, 20]); s = random.randint(1, T - 1)
        beta = random.choice([Fraction(1, 2000), Fraction(1, 100), Fraction(3, 1)])
        m, v = _groups_exact(xs, ws, s, T)
        Wg = exact_greedy(m, v, beta)
        cands = [Fraction(0), Fraction(1)] + [v[k] / m[k] for k in range(s + 1) if m[k] > 0]

        def f(lam):
            return lam * beta + sum(max(vk - lam * mk, Fraction(0)) for vk, mk in zip(v, m))
        # f is convex piecewise linear with breakpoints at the ratios: its min
        # over [0,1] is attained at a candidate
        assert min(f(c) for c in cands if 0 <= c <= 1) == Wg
        assert f(Fraction(1)) == beta and all(f(c) >= Wg for c in cands)


def test_float_welfare_agrees_with_repo():
    """The module's float welfare equals wtm.welfare on a discrete Q."""
    for _ in range(10):
        x = np.sort(RNG.random(5)); w = RNG.dirichlet(np.ones(5))
        prob = DecisionProblem(BetaRisk(1, 8), 12, 0.002, 2)
        dq = DiscreteRisk(tuple(x), tuple(w / w.sum()))
        for s in (1, 4, 9):
            assert abs(float_welfare(prob, s, x, w) - welfare_y(dq, s, 1, 12, 0.002)) < 1e-14


def test_bernstein_bound_rigorous():
    """Lemma 6: the lower bound is <= the true minimum, including narrow dips
    that fall between the points of a fine grid."""
    for _ in range(40):
        deg = random.randint(2, 30)
        c = [Fraction(random.randint(-100, 100), random.randint(1, 50)) for _ in range(deg + 1)]
        L = bernstein_lower_bound(mono_to_bernstein(c))
        xs = np.linspace(0, 1, 20001)
        vals = np.polyval([float(v) for v in c[::-1]], xs)
        assert float(L) <= vals.min() + 1e-9
    # a sharp dip at an irrational point: (p - 1/sqrt 2)^2 - 1e-12 is negative
    # only on an interval of width ~2e-6, missed by a 1001-point grid
    a = Fraction(7071067811865, 10 ** 13)
    c = [a * a - Fraction(1, 10 ** 12), -2 * a, Fraction(1)]
    grid = np.linspace(0, 1, 1001)
    assert (np.polyval([float(v) for v in c[::-1]], grid) > 0).all()
    assert bernstein_lower_bound(mono_to_bernstein(c)) < 0


def _random_members(prob, n, grid):
    """Random members of M_tau on a grid (vertices + interior mixtures)."""
    A = prob.eval_basis(grid) / prob.mu[:, None]
    out = []
    for _ in range(n):
        c = RNG.normal(size=len(grid))
        res = linprog(c, A_eq=A, b_eq=np.ones(len(prob.mu)), bounds=[(0, None)] * len(grid),
                      method="highs")
        out.append(res.x)
    mixes = [0.5 * (out[i] + out[(i + 1) % n]) for i in range(n)]
    return out + mixes


def test_certificates_dominate_members():
    """Lemmas 4-5 end to end: for a certified pair, every sampled member of M
    and every (theta, lambda) in each leaf satisfies G <= U_cert(leaf); and the
    pairwise welfare difference is negative on every sampled member."""
    prob = DecisionProblem(BetaRisk(1, 32), 12, 0.01, 2)
    d, r = 1, 2
    res = bound_pair(prob, d, r, max_boxes=400)
    assert res.status == "certified", res.note
    grid = np.linspace(0, 1, 801)
    members = _random_members(prob, 12, grid)
    PHIr, UPHIr = prob.eval_kernels(r, grid)
    PHId, UPHId = prob.eval_kernels(d, grid)
    checked = 0
    for w in members:
        assert float_welfare(prob, r, grid, w) - float_welfare(prob, d, grid, w) < 0
        mr, vr, md, vd = PHIr @ w, UPHIr @ w, PHId @ w, UPHId @ w
        for box, U, _ in res.leaves:
            for _ in range(3):
                th = float(box[0]) + RNG.random() * float(box[1] - box[0])
                la = float(box[2]) + RNG.random() * float(box[3] - box[2])
                a = np.array(nested_alloc(th, r))
                if a @ mr > prob.beta:
                    continue
                G = a @ vr - la * prob.beta - np.maximum(vd - la * md, 0).sum()
                assert G <= float(U) + 1e-13, (box, G, float(U))
                checked += 1
    assert checked > 100


def test_corrupted_certificate_stays_sound():
    """Corrupting the multipliers can only weaken the bound: the exact checker
    adds the true negative part of R on [0,1], so U_cert stays above the LP
    value of the box (a value attained by a near-member of M)."""
    prob = DecisionProblem(BetaRisk(1, 8), 12, 0.002, 3)
    d, r = 4, 5
    gs = GridState(prob, d, r)
    box = (Fraction(1), Fraction(3, 2), Fraction(1, 4), Fraction(1, 2))
    sol = solve_box_lp(gs, box)
    lp_value = (sol["value"]) * prob.beta         # primal (grid) value, beta units -> abs
    for scale in (1.0, 0.5, 0.9, 1.3):
        for noise in (0.0, 1e-3):
            y = sol["y"] * scale * (1 + noise * RNG.normal(size=len(sol["y"])))
            U, L, _ = verify_box_certificate(prob, d, r, box, y, sol["eta"] * scale,
                                             sol["omega"])
            assert float(U) >= lp_value - 1e-12 * prob.beta, (scale, noise, float(U), lp_value)


def test_exact_witness_is_member():
    """Lemma 8: an exact witness matches every moment exactly and has
    W^r > W^d exactly (Theorem 1's case, Beta(1,32), T=20, beta=0.01, tau=2)."""
    prob = DecisionProblem(BetaRisk(1, 32), 20, 0.01, 2)
    D, W = prob.truth_argmax()
    assert D == [5]
    res = bound_pair(prob, 5, 1, max_boxes=50)
    assert res.status == "witness"
    wit = res.witness
    eps, g = wit["eps"], wit["g"]
    n = prob.n
    for j in range(n + 1):     # E_Q[p^j], computed independently in the monomial basis
        mA = sum((w * a ** j for a, w in zip(wit["atoms"], wit["weights"])), Fraction(0))
        mB = prob.dist.mixed_exact(j, 0) + sum((c * prob.dist.mixed_exact(j + i, 0)
                                                for i, c in enumerate(g)), Fraction(0))
        assert (1 - eps) * mA + eps * mB == prob.dist.mixed_exact(j, 0)
    assert all(w >= 0 for w in wit["weights"])
    assert all(0 <= a <= 1 for a in wit["atoms"])
    assert bernstein_lower_bound(mono_to_bernstein([1 + g[0]] + list(g[1:]))) >= 0
    assert wit["W_r"] > wit["W_d"]


def test_truth_argmax_matches_repo():
    """Exact D*(P) agrees with the repository's float t* on the benchmark cells."""
    from wtm.config import population_cells
    from wtm.welfare import PopulationConfig
    for cell in [c for c in population_cells() if c.gamma == 1]:
        prob = DecisionProblem(cell.dist, cell.T, cell.beta, 2)
        D, _ = prob.truth_argmax()
        assert PopulationConfig(cell.dist, 1, cell.T, cell.beta).t_star == D[0]


def test_actionable_candidate_set():
    """first_date = tau restricts candidates to {tau..T-1}; the exact argmax
    over them agrees with the repository's welfare trajectory."""
    from wtm.welfare import PopulationConfig
    for dist, T, beta in [(BetaRisk(1, 32), 20, 0.01), (BetaRisk(1, 4), 12, 0.002)]:
        W = PopulationConfig(dist, 1, T, beta).W
        for tau in (2, 3, 4):
            prob = DecisionProblem(dist, T, beta, tau, first_date=tau)
            assert prob.dates == list(range(tau, T))
            D, _ = prob.truth_argmax()
            assert D[0] == tau + int(np.argmax(W[tau - 1:]))


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except Exception as exc:          # noqa: BLE001
            failed += 1
            print(f"FAIL  {t.__name__}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return failed


if __name__ == "__main__":
    sys.exit(1 if main() else 0)
