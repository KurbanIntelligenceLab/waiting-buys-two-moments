"""Experiment 9 -- the finite-sample timing-rule benchmark.

Strict paired design: for each (cell, seed) the prefix is simulated **once** and
the identical `PrefixData` is handed to all eight rules.  No rule ever draws its
own data; randomised rules take a separate rule seed derived from the frozen
protocol in `frozen_benchmark.yaml`.  Selected dates are scored against the exact
mean-field trajectory, never against the realised sample, so all reported
variation is sampling variation in the planner's data and not error in the target.

Every run produces one row per rule, including runs where a rule returned no
estimate: `success = False` with a `failure_reason`, scored rather than dropped.

Checkpointing is per cell, so a killed sweep resumes from the last completed
cell rather than restarting.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from wtm.config import config_id, load_frozen, population_cells  # noqa: E402
from wtm.confidence_set import build_region  # noqa: E402
from wtm.lp import IdentifiedSet, support_grid  # noqa: E402
from wtm.moment_projection import (empirical_path_counts,  # noqa: E402
                                   moments_with_covariance)
from wtm.plugin_rules import nnls_plugin, npmle_plugin, parametric_beta  # noqa: E402
from wtm.regret_calibrated import (ENSEMBLE_DEFAULT, LAMBDA_DEFAULT,  # noqa: E402
                                   record_counts_from_prefix, regret_calibrated)
from wtm.rules import evaluate  # noqa: E402
from wtm.simulate import simulate_prefix  # noqa: E402
from wtm.welfare import PopulationConfig  # noqa: E402
from wtm.worst_case import (FROZEN_LOGIC, FROZEN_SET_MODE,  # noqa: E402
                            compute_bounds, decide, decide_envelope)

WORST_CASE_GRID = 201        # verified decision-identical to G=501 (6/6)

# Table 4 scope, frozen 2026-08-27: gamma = 1 only.
# The moment-based rules (parametric Beta, both worst-case rules) are defined
# through the ordinary-moment representation of Lemma 1, which holds only at
# gamma = 1: there the observable functionals reach degree exactly 2t-1, so the
# 2t moments are exactly identified.  At gamma = 2 or 4 they reach degree
# gamma*t + t - 1, and "the first 2t moments" is no longer what the data
# determine.  Pooling those cells into Table 4 would compare rules that are not
# all well defined on them.  Robustness to gamma != 1 belongs in Experiment 13,
# restricted to the rules that fit the gamma-specific record likelihood
# directly (NPMLE, NNLS, regret-calibrated).
TABLE4_GAMMA = 1
MOMENT_BASED_RULES = ("parametric_beta", "worst_case_envelope", "worst_case_lp")
RULES = ["oracle", "act_immediately", "parametric_beta", "nnls", "npmle",
         "worst_case_envelope", "worst_case_lp", "regret_calibrated"]


class _SkipMomentRule(Exception):
    """gamma != 1: the ordinary-moment representation is not defined."""


class _NoBounds(Exception):
    """The identified set was empty, so no bounds are available."""


def seeds_for(cell_id: str, seed_list):
    """Frozen seed protocol: simulation and rule streams, both recorded."""
    base = int(cell_id, 16)
    for s in seed_list:
        sim = (base + int(s)) % (2 ** 32)
        rule = (sim + 2 ** 31 - 1) % (2 ** 32)
        yield int(s), sim, rule


def run_cell(args):
    cell, N, t, seed_list = args
    pop = PopulationConfig(cell.dist, cell.gamma, cell.T, cell.beta)
    cell_id = config_id(dist=cell.dist_name, gamma=cell.gamma, T=cell.T,
                        beta=cell.beta, N=N, t=t)
    rows = []

    for seed_index, sim_seed, rule_seed in seeds_for(cell_id, seed_list):
        data = simulate_prefix(cell.dist, N, cell.T, cell.gamma, t, seed=sim_seed)
        rec = record_counts_from_prefix(data, t)
        m_hat, cov = moments_with_covariance(empirical_path_counts(data, t),
                                             t, cell.gamma)

        outcomes = {}

        outcomes["oracle"] = (pop.t_star, True, "", 0)
        outcomes["act_immediately"] = (1, True, "", 0)

        moment_ok = (cell.gamma == TABLE4_GAMMA)
        if not moment_ok:
            for r in MOMENT_BASED_RULES:
                outcomes[r] = (-1, False,
                               "moment_representation_undefined_for_gamma_ne_1", 0.0)

        plugin_specs = (("parametric_beta",
                          lambda: parametric_beta(rec, cell.T, cell.gamma, t,
                                                  cell.beta, moments=m_hat)),
                         ("nnls",
                          lambda: nnls_plugin(rec, cell.T, cell.gamma, t, cell.beta)),
                         ("npmle",
                          lambda: npmle_plugin(rec, cell.T, cell.gamma, t, cell.beta)))
        for name, fn in plugin_specs:
            if name in MOMENT_BASED_RULES and not moment_ok:
                continue
            t0 = time.perf_counter()
            try:
                o = fn()
                outcomes[name] = (o.selected_date, o.success, o.failure_reason,
                                  time.perf_counter() - t0)
            except Exception as exc:                     # noqa: BLE001
                outcomes[name] = (-1, False, f"exception:{type(exc).__name__}",
                                  time.perf_counter() - t0)

        # both worst-case rules share one set and one set of lower bounds
        t0 = time.perf_counter()
        try:
            if not moment_ok:
                raise _SkipMomentRule
            b = compute_bounds(m_hat, t, cell.gamma, cell.T, cell.beta,
                               grid_size=WORST_CASE_GRID, cov=cov,
                               set_mode=FROZEN_SET_MODE, lazy_upper=True)
            d_lp = decide(b, FROZEN_LOGIC)
            outcomes["worst_case_lp"] = (d_lp["selected_date"], d_lp["success"],
                                         d_lp.get("failure_reason", ""),
                                         time.perf_counter() - t0)
        except _SkipMomentRule:
            b = None
        except Exception as exc:                         # noqa: BLE001
            b = None
            outcomes["worst_case_lp"] = (-1, False, f"exception:{type(exc).__name__}",
                                         time.perf_counter() - t0)

        t0 = time.perf_counter()
        try:
            if not moment_ok:
                raise _SkipMomentRule
            if b is None or not b.feasible:
                raise _NoBounds
            region = build_region(m_hat, cov, t)
            S = IdentifiedSet.from_confidence_region(
                region, t, support_grid(WORST_CASE_GRID))
            d_env = decide_envelope(S, b.dates, b.lower, cell.gamma, cell.T,
                                    cell.beta)
            outcomes["worst_case_envelope"] = (d_env["selected_date"], True, "",
                                               time.perf_counter() - t0)
        except _SkipMomentRule:
            pass
        except _NoBounds:
            outcomes["worst_case_envelope"] = (
                -1, False, "identified set empty (no bounds from the LP rule)",
                time.perf_counter() - t0)
        except Exception as exc:                         # noqa: BLE001
            outcomes["worst_case_envelope"] = (-1, False,
                                               f"exception:{type(exc).__name__}",
                                               time.perf_counter() - t0)

        t0 = time.perf_counter()
        try:
            o = regret_calibrated(rec, cell.T, cell.gamma, t, cell.beta,
                                  rule_seed=rule_seed)
            outcomes["regret_calibrated"] = (o.selected_date, o.success,
                                             o.failure_reason,
                                             time.perf_counter() - t0)
        except Exception as exc:                         # noqa: BLE001
            outcomes["regret_calibrated"] = (-1, False,
                                             f"exception:{type(exc).__name__}",
                                             time.perf_counter() - t0)

        for rule in RULES:
            date, ok, why, secs = outcomes[rule]
            row = {
                "cell_id": cell_id, **cell.as_row(), "N": N, "prefix_t": t,
                "seed_index": seed_index, "simulation_seed": sim_seed,
                "rule_seed": rule_seed, "rule": rule,
                "success": bool(ok), "failure_reason": why, "runtime_s": secs,
                "true_optimal_date": pop.t_star, "optimal_welfare": pop.W_star,
                "stakes": pop.stakes,
                "worst_case_grid": WORST_CASE_GRID,
                "set_mode": FROZEN_SET_MODE, "logic": FROZEN_LOGIC,
                "ensemble_size": ENSEMBLE_DEFAULT, "lambda": LAMBDA_DEFAULT,
            }
            if ok:
                row.update(evaluate(date, pop))
            else:
                row.update({"selected_date": -1,
                            "true_optimal_date": pop.t_star,
                            "exact_argmax_match": False,
                            "true_welfare_at_selected": np.nan,
                            "optimal_welfare": pop.W_star,
                            "welfare_efficiency": np.nan,
                            "relative_regret": np.nan, "squared_regret": np.nan})
            rows.append(row)
    return rows


def build_tasks(sizes, prefixes, cfg=None, gammas=None):
    """Tasks for the benchmark.  `gammas` defaults to the frozen Table 4 scope."""
    cfg = cfg or load_frozen()
    seed_list = cfg["seeds"]["list"]
    if not seed_list:
        raise RuntimeError("seed list is not frozen; see NEEDS_DECISION D3")
    gammas = set(gammas) if gammas else {TABLE4_GAMMA}
    return [(cell, int(N), int(t), seed_list)
            for cell in population_cells(cfg) if cell.gamma in gammas
            for N in sizes for t in prefixes]
