"""CMS and the observation window (results/audit/cms_window_audit.json,
results/audit/cms_equivalence_audit.json).

The path cells separate period-tau exits from survivors. For the idealized
procedure (bootstrap randomness independent of the data) the two resampling
modes give the recommendation the same distribution: (i) multinomial
aggregation, checked exactly on a small case; (ii) the rule sees a draw only
through `draw_risks`, and there only through `moments_with_covariance`,
checked on the source; (iii) the per-draw branches are merge-invariant.
They are not seed-identical. Path resampling must still reproduce the
archived Table 3 draws.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from wtm import cms  # noqa: E402
from wtm.config import config_id, load_frozen, population_cells  # noqa: E402
from wtm.moment_cone import build_path_alphabet, cohort_design  # noqa: E402
from wtm.moment_projection import empirical_path_counts, moments_with_covariance  # noqa: E402
from wtm.simulate import simulate_prefix  # noqa: E402


def data_for(dist="beta_1_8", T=12, beta=0.002, N=1000, seed=1001, tau=2):
    c = next(c for c in population_cells(load_frozen())
             if c.gamma == 1 and c.dist_name == dist and c.T == T and c.beta == beta)
    cid = config_id(dist=dist, gamma=1, T=T, beta=beta, N=N, t=tau)
    d = simulate_prefix(c.dist, N, T, 1, tau, seed=(int(cid, 16) + seed) % 2 ** 32)
    return cid, np.asarray(empirical_path_counts(d, tau), float)


def resplit(counts, tau, frac):
    """Same pre-failure observations, a different period-tau exit/survivor split."""
    pa = build_path_alphabet(tau)
    idx = {p: i for i, p in enumerate(pa.paths)}
    c = counts.copy()
    for (e, obs), i in idx.items():
        if e == tau:
            j = idx[(None, obs)]
            tot = c[i] + c[j]
            c[i] = np.floor(frac * tot)
            c[j] = tot - c[i]
    return c


@pytest.mark.parametrize("tau", [2, 3, 4])
def test_moment_map_ignores_the_split(tau):
    pa = build_path_alphabet(tau)
    B = np.linalg.pinv(cohort_design(tau, 1)) @ pa.to_cohort.T
    idx = {p: i for i, p in enumerate(pa.paths)}
    for (e, obs), i in idx.items():
        if e == tau:
            assert np.array_equal(B[:, i], B[:, idx[(None, obs)]])


def test_prefailure_counts_ignore_the_split():
    _, c = data_for()
    for frac in (0.0, 0.3, 1.0):
        assert np.array_equal(cms.prefailure_counts(c, 2), cms.prefailure_counts(resplit(c, 2, frac), 2))


def test_path_resampling_draws_depend_on_the_split_prefailure_draws_do_not():
    _, c = data_for()
    alt = resplit(c, 2, 0.0)
    draws = lambda x, s: cms.prefailure_counts(  # noqa: E731
        np.random.default_rng(s).multinomial(int(x.sum()), x / x.sum()).astype(float), 2)
    # path resampling: the pre-failure content of a draw changes with the split
    assert any(not np.array_equal(draws(c, s), draws(alt, s)) for s in range(5))
    # pre-failure resampling: identical
    pc, pa_ = cms.prefailure_counts(c, 2), cms.prefailure_counts(alt, 2)
    for s in range(5):
        assert np.array_equal(draws(pc, s), draws(pa_, s))
        m1, _ = moments_with_covariance(draws(pc, s), 2, 1)
        m2, _ = moments_with_covariance(draws(pa_, s), 2, 1)
        assert np.array_equal(m1, m2)


def test_default_is_the_manuscript_implementation():
    import inspect
    assert inspect.signature(cms.bootstrap_rule).parameters["resampling"].default == "paths"


def test_path_resampling_reproduces_an_archived_draw():
    """One archived Table 3 draw, recomputed with resampling='paths' (~20 s)."""
    from experiments.scripts.cms_table2_analyze import draw_records
    matches = [rec for rec in draw_records()
               if (rec["distribution"], rec["T"], rec["beta"], rec["N"],
                   rec["prefix_t"], rec["seed_index"], rec["draw"])
               == ("beta_1_8", 12, 0.002, 1000, 2, 1001, 0)]
    assert len(matches) == 1, "expected exactly one archived CMS draw"
    rec = matches[0]
    cid, c = data_for()
    r = cms.bootstrap_rule(c, 2, 1, 12, 0.002, n_boot=1, grid_size=401, resampling="paths",
                           seed=(int(cid, 16) + 1001 + 0) % 2 ** 32)
    want = json.loads(rec["max_risk"])
    assert {str(k): v for k, v in r.per_draw[0].items()} == pytest.approx(want, rel=1e-9, abs=1e-12)


def test_multinomial_aggregation_exact():
    """(i): resample fine cells then merge == merge then resample, as exact laws."""
    from fractions import Fraction
    from itertools import product
    from math import factorial

    def pmf(n, p):
        out = {}
        for x in product(range(n + 1), repeat=len(p)):
            if sum(x) != n:
                continue
            w = Fraction(factorial(n))
            for xi, pi in zip(x, p):
                w = w / factorial(xi) * pi ** xi
            out[x] = w
        return out

    p = [Fraction(1, 10), Fraction(2, 10), Fraction(3, 10), Fraction(4, 10)]
    pairs = [(0, 1)]                          # cell 0 (exit at tau) merges into cell 1
    for n in (1, 3, 5):
        fine = pmf(n, p)
        merged = {}
        for x, w in fine.items():
            y = (x[0] + x[1], x[2], x[3])
            merged[y] = merged.get(y, 0) + w
        assert merged == pmf(n, [p[0] + p[1], p[2], p[3]])


def _names(fn):
    import ast
    import inspect
    import textwrap
    return ast.parse(textwrap.dedent(inspect.getsource(fn)))


def test_draw_enters_only_through_draw_risks_and_moments():
    """(ii): code-level dependency. In bootstrap_rule, `draw` is produced by the
    multinomial and consumed only by draw_risks; in draw_risks it is consumed
    only by moments_with_covariance."""
    import ast

    def consumers(fn, name):
        tree = _names(fn)
        uses = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                for arg in list(node.args) + [k.value for k in node.keywords]:
                    if isinstance(arg, ast.Name) and arg.id == name:
                        f = node.func
                        uses.append(f.id if isinstance(f, ast.Name) else f.attr)
        loads = [n for n in ast.walk(tree)
                 if isinstance(n, ast.Name) and n.id == name and isinstance(n.ctx, ast.Load)]
        return uses, len(loads)

    uses, loads = consumers(cms.bootstrap_rule, "draw")
    assert uses == ["draw_risks"] and loads == 1
    uses, loads = consumers(cms.draw_risks, "draw")
    assert uses == ["moments_with_covariance"] and loads == 1


def test_draw_risks_branches_merge_invariant():
    """(iii): fine and merged draws give the same moments up to rounding and the
    same feasibility / projection / singleton branches."""
    from wtm.lp import IdentifiedSet, support_grid
    from wtm.moment_projection import project_to_cone
    grid = support_grid(401)
    _, c = data_for()
    for s in range(10):
        f = np.random.default_rng(s).multinomial(int(c.sum()), c / c.sum()).astype(float)
        out = []
        for x in (f, cms.prefailure_counts(f, 2)):
            m, cov = moments_with_covariance(x, 2, 1)
            feas = IdentifiedSet.from_moments(m, 2, grid).feasible_point()[0] is not None
            used = m if feas else project_to_cone(m, cov, grid, 2).moments
            out.append((m, cov, feas, cms.on_cone_boundary(used, 2, tol=1e-9)[0]))
        (m1, c1, f1, p1), (m2, c2, f2, p2) = out
        assert np.allclose(m1, m2, rtol=0, atol=1e-15) and np.allclose(c1, c2, rtol=0, atol=1e-18)
        assert (f1, p1) == (f2, p2)


def test_modes_are_not_seed_identical():
    """Same seed, different draws: the merged category consumes different random
    numbers. This is a seed-level difference, not an information difference."""
    _, c = data_for()
    m = cms.prefailure_counts(c, 2)
    differ = 0
    for s in range(5):
        a = cms.prefailure_counts(np.random.default_rng(s).multinomial(int(c.sum()), c / c.sum()), 2)
        b = np.random.default_rng(s).multinomial(int(m.sum()), m / m.sum())
        differ += not np.array_equal(a, b)
    assert differ > 0


def test_legacy_seed_scheme_is_the_archived_one():
    assert cms.draw_seed("legacy", 5, 1001, 3) == 1009
    assert cms.draw_seed("legacy", 2 ** 32 - 1, 1001, 0) == 1000


def test_independent_streams_are_distinct_from_data_and_rule_streams():
    """Every independent draw stream differs from every other draw stream and from
    every integer-seeded data or rule stream of the same cell."""
    cid, _ = data_for()
    base = int(cid, 16)
    states = {}
    def key(seed):
        return tuple(np.random.default_rng(seed).bit_generator.random_raw(4))
    for s in range(1001, 1021):
        sim = (base + s) % 2 ** 32
        states[("data", s)] = key(sim)
        states[("rule", s)] = key((sim + 2 ** 31 - 1) % 2 ** 32)
        for d in range(20):
            states[("draw", s, d)] = key(cms.draw_seed("independent", base, s, d))
    assert len(set(states.values())) == len(states)
    # and the legacy scheme is exactly what collides
    assert key(cms.draw_seed("legacy", base, 1001, 0)) == states[("data", 1001)]


def test_new_schemes_never_write_to_the_archive():
    sys.path.insert(0, str(ROOT / "experiments" / "scripts"))
    from experiments.scripts.cms_table2_run import CKPT, ckpt_dir
    assert ckpt_dir("paths", "legacy") == CKPT
    others = {ckpt_dir(r, g) for r in ("paths", "prefailure") for g in ("legacy", "independent")
              if (r, g) != ("paths", "legacy")}
    assert CKPT not in others and len(others) == 3
