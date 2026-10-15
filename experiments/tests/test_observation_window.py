"""Regression tests pinning the length-t observation window (R15 correction).

These exist because the implementation and the paper disagreed: the code used to
enumerate exits through the end of period t and to use phi_{t,k}*(1-p) for the
terminal cell, giving kernels of degree 2t and exact rank 2t+1 -- one moment more
than Lemma 1 allows.  The corrected window stops the exit record at t-1 and uses
the plain cohort kernel phi_{t,k} at t.

Seven of the nine tests below fail against the old Window-B model (verified, not
assumed).  The other two are well-formedness invariants that hold for any sane
window and are marked as such: `test_record_kernels_sum_to_one`, because both
windows are proper probability distributions, and
`test_empirical_counts_align_with_kernels`, which is deliberately insensitive to
*which* window is in force -- its job is to catch the two definition sites
drifting apart, which it does.

Run with:  python -m pytest experiments/tests/test_observation_window.py -q
      or:  python experiments/tests/test_observation_window.py
"""

from __future__ import annotations

import sys
from fractions import Fraction
from pathlib import Path

import numpy as np
from scipy.optimize import linprog

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from wtm.distributions import BetaRisk
from wtm.lp import IdentifiedSet, support_grid
from wtm.observation import (cohort_functionals, individual_functionals, phi,
                             phi_exit)
from wtm.regret_calibrated import record_counts_from_prefix
from wtm.simulate import simulate_prefix

MAX_T = 10


def _exact_rank(polys) -> int:
    """Rank over Q of the monomial coefficient vectors -- no floating point."""
    width = max(p.degree for p in polys) + 1
    rows = []
    for p in polys:
        v = list(p.to_monomial())
        rows.append([Fraction(int(x)) for x in v] + [Fraction(0)] * (width - len(v)))
    r = 0
    for c in range(width):
        piv = next((i for i in range(r, len(rows)) if rows[i][c] != 0), None)
        if piv is None:
            continue
        rows[r], rows[piv] = rows[piv], rows[r]
        inv = rows[r][c]
        rows[r] = [x / inv for x in rows[r]]
        for i in range(len(rows)):
            if i != r and rows[i][c] != 0:
                f = rows[i][c]
                rows[i] = [a - f * b for a, b in zip(rows[i], rows[r])]
        r += 1
    return r


# -- 1. the span is exactly Pi_{2t-1} ----------------------------------------
def test_record_kernels_have_degree_at_most_2t_minus_1():
    for t in range(1, MAX_T + 1):
        for key, f in individual_functionals(t):
            assert f.degree <= 2 * t - 1, (
                f"t={t} atom {key} has degree {f.degree} > 2t-1={2 * t - 1}; "
                "the Window-B model has been reintroduced")


def test_record_family_has_exact_rank_2t():
    for t in range(1, MAX_T + 1):
        r = _exact_rank([f for _, f in individual_functionals(t)])
        assert r == 2 * t, f"t={t}: exact rank {r}, expected 2t={2 * t}"


def test_record_span_equals_cohort_span():
    """Lemma 1: the record law spans the same Pi_{2t-1} as the cohort counts."""
    for t in range(1, MAX_T + 1):
        rec = [f for _, f in individual_functionals(t)]
        coh = [f for _, f in cohort_functionals(t)]
        assert _exact_rank(rec) == _exact_rank(coh) == 2 * t
        # the union cannot exceed either, i.e. neither adds a direction
        assert _exact_rank(rec + coh) == 2 * t, (
            f"t={t}: records and cohorts together span more than Pi_2t-1")


def test_terminal_cell_is_the_plain_cohort_kernel():
    """Guards the specific regression: phi_{t,k}, never phi_{t,k}*(1-p)."""
    for t in range(1, MAX_T + 1):
        terminal = [(key, f) for key, f in individual_functionals(t)
                    if key[0] != "exit"]
        assert len(terminal) == t + 1
        for (_, _, k), f in terminal:
            assert f.coeffs == phi(t, k).coeffs, (
                f"t={t}, k={k}: terminal kernel is not phi_(t,k)")


def test_exit_records_stop_at_t_minus_1():
    for t in range(1, MAX_T + 1):
        exits = [key for key, _ in individual_functionals(t) if key[0] == "exit"]
        assert all(s <= t - 1 for _, s, _ in exits), (
            f"t={t}: window contains an exit at period t, which is only "
            "observable at t+1")
        assert len(exits) == sum(s + 1 for s in range(1, t))


# -- 2. the atoms are a probability distribution ------------------------------
def test_record_kernels_sum_to_one():
    """Columns of the NPMLE design matrix must sum to 1 at every p.

    Well-formedness only: Window B also summed to 1, so this does not
    discriminate between the two models.  It catches a malformed window.
    """
    grid = np.linspace(0.0, 1.0, 401)
    for t in range(1, MAX_T + 1):
        total = np.zeros_like(grid)
        for _, f in individual_functionals(t):
            total += f.evaluate(grid)
        assert np.allclose(total, 1.0, atol=1e-12), (
            f"t={t}: record atoms sum to {total.min()}..{total.max()}, not 1")


# -- 3. moment-matched pairs are observationally identical --------------------
def _extreme_pair(dist, T, t, grid):
    """The paper's extreme-pair construction for the horizon-survival functional."""
    S = IdentifiedSet(dist, t, grid)
    c = (1.0 - grid) ** (T - 1)
    lo = linprog(c=c, A_eq=S.A, b_eq=S.b, bounds=[(0.0, None)] * S.n, method="highs")
    hi = linprog(c=-c, A_eq=S.A, b_eq=S.b, bounds=[(0.0, None)] * S.n, method="highs")
    assert lo.success and hi.success
    return lo.x, hi.x


def test_moment_matched_pairs_give_identical_records():
    """Theorem 1's premise: matching m_0..m_{2t-1} must equalise the record law.

    Under the old window these pairs differed by ~9e-5 and were separable at
    N ~ 9e4 -- inside the benchmark's own grid.
    """
    grid = support_grid(2001)
    for dist, T, t in [(BetaRisk(1, 32), 20, 2),
                       (BetaRisk(1, 8), 20, 2),
                       (BetaRisk(1, 32), 20, 3),
                       (BetaRisk(1, 4), 12, 2)]:
        w0, w1 = _extreme_pair(dist, T, t, grid)
        worst = 0.0
        for key, f in individual_functionals(t):
            vals = f.evaluate(grid)
            worst = max(worst, abs(float(vals @ w0) - float(vals @ w1)))
        assert worst < 1e-12, (
            f"{dist}, T={T}, t={t}: moment-matched pair separated by {worst:.2e} "
            "in the record law; the window identifies a moment beyond 2t-1")


# -- 4. simulated counts align atom-by-atom with the kernels ------------------
def test_empirical_counts_align_with_kernels():
    """Counts must use the same atoms, in the same order, and total N.

    This is a *consistency* test, not a window discriminator: it passes for any
    window whose two definition sites agree.  What it catches is drift between
    them -- patching only one side trips the length check immediately.
    """
    N = 200_000
    for dist, T, t in [(BetaRisk(1, 8), 20, 2), (BetaRisk(1, 4), 20, 3),
                       (BetaRisk(1, 32), 12, 4)]:
        data = simulate_prefix(dist, N=N, T=T, gamma=1, t=t, seed=20260916)
        counts = record_counts_from_prefix(data, t)
        atoms = individual_functionals(t)

        assert len(counts) == len(atoms), (
            f"t={t}: {len(counts)} counts vs {len(atoms)} kernels -- "
            "record_counts_from_prefix and individual_functionals are out of step")
        assert counts.sum() == N, (
            f"t={t}: counts total {counts.sum():.0f}, not N={N}; the window is "
            "dropping or double-counting individuals")

        # every simulated frequency must match its kernel's population value
        expected = np.array([f.expectation(dist) for _, f in atoms])
        observed = counts / N
        se = np.sqrt(np.maximum(expected * (1 - expected), 1e-12) / N)
        z = np.abs(observed - expected) / se
        assert z.max() < 6.0, (
            f"t={t}: atom {atoms[int(z.argmax())][0]} is {z.max():.1f} SE from "
            "its kernel -- counts and kernels describe different windows")


def test_period_t_exit_does_not_change_the_record():
    """The decisive behavioural check: the period-t exit outcome is unobserved.

    Individuals who exit at the end of period t must still be counted in the
    period-t cohort, exactly like those who survive it.
    """
    N = 50_000
    t = 3
    data = simulate_prefix(BetaRisk(1, 8), N=N, T=20, gamma=1, t=t, seed=7)
    counts = record_counts_from_prefix(data, t)

    n_exit_at_t = int(np.sum(data.exited_by == t))
    assert n_exit_at_t > 0, "test is vacuous: nobody exited at period t"

    terminal = counts[-(t + 1):]
    n_active_at_t = int(np.sum(data.active_mask[t - 1]))
    assert terminal.sum() == n_active_at_t, (
        "terminal cell does not equal the pool active at the start of period t")
    assert terminal.sum() == int(np.sum(data.exited_by == -1)) + n_exit_at_t, (
        "period-t exiters were excluded from the terminal cell -- that is the "
        "Window-B model")


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
