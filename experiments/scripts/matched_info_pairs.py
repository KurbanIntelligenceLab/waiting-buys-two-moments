"""Matched-information evaluation on compatible populations (Appendix D.3).

VERSION: reconstruction-v1. Written from the text of Appendix D.3 of the
2026-09-30 manuscript, because the original D.3 code is not in the repository.
It does NOT reproduce the PDF's Tables 6-7 (paper/reconciliation/README.md,
section 03). Treat its output as a separate version of the experiment.

Ten non-CMS procedures are scored under BOTH members of the 30 exact tau = 2
extreme pairs behind Table 1 (`extreme_pair_exact.py`), including the pair with
zero regret separation.

Both members of a pair induce identical probabilities for every observable
path through period-2 warnings, before period-2 failures (Lemma 1), so one
dataset drawn from that shared law is a valid draw under either member. It is
simulated from the lower member. Every procedure receives that one dataset;
none receives period-2 failures or the true population. Each selected date is
scored against both members' exact welfare curves, over dates 1..T-1.

Settings (Appendix D.3):
  NPMLE / EM        common-window likelihood, uniform 200-point grid,
                    uniform initial weights, NPMLE tolerance 1e-8
  bagging / RC      one shared ensemble of 40 attempted bootstrap fits
  moment rules      cohort moments and covariance from the same window
  LP rules          101-point grid, 9 allocation thresholds, 11 dual prices,
                    one 5-point refinement round
  trials            N in {1e3, 1e5}, seeds 1001..1005 (300 planned datasets)

Usage:
  python experiments/scripts/matched_info_pairs.py                     # publication settings -> runs/matched_info_pairs/
  python experiments/scripts/matched_info_pairs.py --out results --force   # replace the canonical archive
  python experiments/scripts/matched_info_pairs.py --smoke --out DIR       # 2 pairs, N=1e3, 1 seed, 5 bootstrap fits

Flags: --out DIR, --workers N, --seeds 1001,1002, --ns 1000,100000,
--n-boot B, --pairs K (first K pairs), --smoke. The publication settings are
the module constants; any override is recorded in the manifest as
non-publication. The default output directory is never results/, and an
existing raw file is never overwritten without --force.

Checkpoints (<out>/matched_info_pairs_cells/) carry a fingerprint of the
source, configuration, observation window, seed policy, settings and pair
definitions. A checkpoint is reused only if its fingerprint matches and it has
exactly the expected rows; otherwise the run stops (or, with --discard-stale,
recomputes the cell).

Writes  <out>/matched_info_pairs_raw.csv (one row per dataset x procedure)
        <out>/matched_info_pairs_manifest.json
Then    python experiments/scripts/matched_info_pairs_analyze.py --raw <out>/matched_info_pairs_raw.csv
"""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
RESULTS = ROOT / "results"
DEFAULT_OUT = ROOT / "runs" / "matched_info_pairs"
VERSION = "reconstruction-v1"
SEED_POLICY = ("sim_seed = (int(config_id(dist='pair_<config_id>', gamma=1, T, beta, N, tau=2), 16)"
               " + seed) mod 2^32; rule_seed = (sim_seed + 2^31 - 1) mod 2^32; data drawn from the"
               " lower pair member; bootstrap = default_rng(rule_seed).multinomial, n_boot draws")

TAU = 2
NS = (1000, 100000)
SEEDS = tuple(range(1001, 1006))
BBOOT = 40
DELTA = 1e-8
EM_ITERS = (30, 55, 80)
LP_GRID = 101
LP_SCAN = dict(n_fill=9, n_lambda=11, n_refine=5, n_rounds=1)
ENV_SCAN = dict(n_coarse=9, n_refine=5, n_rounds=1)

PROCEDURES = ("certified_npmle", "em_30", "em_55", "em_80", "bagged_npmle",
              "regret_calibrated", "parametric_beta", "achieved_welfare_lp",
              "worst_case_envelope", "maximin_welfare")


def exact_pairs():
    """The 30 tau = 2 pairs: exact members and their exact relative regrets."""
    from fractions import Fraction  # noqa: F401  (members are Fractions)
    from wtm.config import config_id, load_frozen, population_cells
    from wtm.decision_id import DecisionProblem
    from wtm.lp import IdentifiedSet, support_grid
    # extreme_pair_rebuild reads its own prefix from sys.argv at import time
    argv, sys.argv = sys.argv, sys.argv[:1]
    try:
        from experiments.scripts.extreme_pair_exact import exact_member, exact_trajectory
        from experiments.scripts.extreme_pair_rebuild import GRID_SIZE, extreme_members
    finally:
        sys.argv = argv

    cells = [c for c in population_cells(load_frozen()) if c.gamma == 1]
    assert len(cells) == 30
    grid = support_grid(GRID_SIZE)
    out = []
    for cell in cells:
        prob = DecisionProblem(cell.dist, cell.T, cell.beta, TAU)
        S = IdentifiedSet(cell.dist, TAU, grid)
        lo, hi = extreme_members(grid, S, cell.T)
        assert lo.success and hi.success
        members = [exact_member(prob, lo.x), exact_member(prob, hi.x)]
        assert all(wi >= 0 for _, w in members for wi in w)
        regrets = []
        for atoms, w in members:
            W = exact_trajectory(prob, atoms, w)
            best = max(W.values())
            regrets.append({s: float(1 - W[s] / best) for s in prob.dates})
        zeta = min(regrets[0][s] + regrets[1][s] for s in prob.dates)
        out.append({
            "config_id": config_id(dist=cell.dist_name, gamma=1, T=cell.T, beta=cell.beta),
            "distribution": cell.dist_name, "T": cell.T, "beta": cell.beta,
            "members": [([float(a) for a in atoms], [float(x) for x in w])
                        for atoms, w in members],
            "regret": regrets, "zeta_over_2": zeta / 2,
        })
    return out


def settings_of(seeds=SEEDS, n_boot=BBOOT) -> dict:
    return {"version": VERSION, "tau": TAU, "seeds": list(seeds), "n_boot": int(n_boot),
            "delta": DELTA, "em_iters": list(EM_ITERS), "lp_grid": LP_GRID, "lp_scan": LP_SCAN,
            "envelope_scan": ENV_SCAN, "procedures": list(PROCEDURES), "seed_policy": SEED_POLICY}


def fingerprint(pair, N, settings) -> str:
    """Everything a cell's rows depend on."""
    from wtm.provenance import CONFIG, observation_window, sha256_file
    code = [Path(__file__)] + sorted((ROOT / "wtm").glob("*.py")) + [
        ROOT / "experiments" / "scripts" / "extreme_pair_exact.py",
        ROOT / "experiments" / "scripts" / "extreme_pair_rebuild.py"]
    h = hashlib.sha256()
    for p in code:
        h.update(f"{p.relative_to(ROOT)}\0{sha256_file(p)}\n".encode())
    h.update(sha256_file(CONFIG).encode())
    h.update(observation_window().encode())
    h.update(json.dumps(settings, sort_keys=True).encode())
    h.update(json.dumps({"pair": {k: pair[k] for k in ("config_id", "T", "beta", "members")},
                         "N": N}, sort_keys=True).encode())
    return h.hexdigest()


class StaleCheckpoint(RuntimeError):
    pass


def load_checkpoint(ck: Path, fp: str, n_rows: int):
    """The checkpoint's path if valid; None if absent; StaleCheckpoint if not reusable."""
    meta = ck.with_suffix(".json")
    if not ck.exists() and not meta.exists():
        return None
    if not (ck.exists() and meta.exists()):
        raise StaleCheckpoint(f"{ck.name}: incomplete checkpoint (data or fingerprint missing)")
    m = json.loads(meta.read_text())
    if m.get("fingerprint") != fp:
        raise StaleCheckpoint(f"{ck.name}: fingerprint differs (code, config, window, seeds, "
                              "settings or pair changed since it was written)")
    df = pd.read_csv(ck)
    if len(df) != n_rows or m.get("rows") != n_rows or df.duplicated(["seed_index", "procedure"]).any():
        raise StaleCheckpoint(f"{ck.name}: {len(df)} rows, expected {n_rows} unique")
    return str(ck)


def run_cell(args):
    pair, N, settings, ckpt, discard_stale = args
    ckpt = Path(ckpt)
    ckpt.mkdir(parents=True, exist_ok=True)
    ck = ckpt / f"{pair['config_id']}_{N}.csv"
    fp = fingerprint(pair, N, settings)
    n_rows = len(settings["seeds"]) * len(PROCEDURES)
    try:
        got = load_checkpoint(ck, fp, n_rows)
        if got:
            return got
    except StaleCheckpoint:
        if not discard_stale:
            raise
        ck.unlink(missing_ok=True)
        ck.with_suffix(".json").unlink(missing_ok=True)
    SEEDS_, BBOOT_ = settings["seeds"], settings["n_boot"]

    from wtm.config import config_id
    from wtm.distributions import DiscreteRisk
    from wtm.lp import IdentifiedSet, group_matrices, support_grid
    from wtm.confidence_set import build_region
    from wtm.moment_projection import empirical_path_counts, moments_with_covariance
    from wtm.npmle import fit_npmle, fit_npmle_certified, record_matrix
    from wtm.plugin_rules import parametric_beta
    from wtm.regret_calibrated import (NPMLE_GRID, _trajectory, expected_regret_rule,
                                       record_counts_from_prefix)
    from wtm.simulate import simulate_prefix
    from wtm.worst_case import FROZEN_SET_MODE, compute_bounds, decide, decide_envelope

    T, beta = pair["T"], pair["beta"]
    atoms, w = pair["members"][0]
    truth = DiscreteRisk(points=tuple(atoms), weights=tuple(np.asarray(w) / sum(w)),
                         name=f"pair_lower_{pair['config_id']}")
    cell_id = config_id(dist=f"pair_{pair['config_id']}", gamma=1, T=T, beta=beta,
                        N=N, t=TAU)
    base = int(cell_id, 16)
    grid = np.linspace(0.0, 1.0, NPMLE_GRID)
    Psi = record_matrix(grid, TAU, 1)
    n_atoms = Psi.shape[0]
    cache = [group_matrices(grid, int(s), 1, T) for s in range(1, T)]
    w0 = np.full(grid.shape, 1.0 / grid.size)

    def argmax_date(W):
        W = np.asarray(W, dtype=float)
        return int(np.argmax(W)) + 1 if np.all(np.isfinite(W)) else -1

    rows = []
    for seed in SEEDS_:
        sim_seed = (base + seed) % 2 ** 32
        rule_seed = (sim_seed + 2 ** 31 - 1) % 2 ** 32
        data = simulate_prefix(truth, N, T, 1, TAU, seed=sim_seed)
        n = np.asarray(record_counts_from_prefix(data, TAU), dtype=float)
        m_hat, cov = moments_with_covariance(empirical_path_counts(data, TAU), TAU, 1)
        dec = {}

        def run(name, fn):
            t0 = time.perf_counter()
            try:
                d, reason = fn()
            except Exception as exc:                          # noqa: BLE001
                d, reason = -1, f"exception:{type(exc).__name__}: {exc}"
            dec[name] = (int(d), reason, time.perf_counter() - t0)

        def npmle():
            fit, _ = fit_npmle_certified(n, grid, TAU, 1, delta=DELTA, Psi=Psi)
            if fit.n_support > n_atoms or not np.isfinite(fit.loglik):
                return -1, "invalid_fit"
            return argmax_date(_trajectory(fit.weights, cache, beta)), ""
        run("certified_npmle", npmle)

        for K in EM_ITERS:
            def em(K=K):
                fit = fit_npmle(n, grid, TAU, 1, tol=1e-300, max_iter=K, Psi=Psi, w0=w0)
                if not np.isfinite(fit.loglik):
                    return -1, "non_finite_loglik"
                return argmax_date(_trajectory(fit.weights, cache, beta)), ""
            run(f"em_{K}", em)

        # bagging and RC share one ensemble; it is built once, timed once
        t0 = time.perf_counter()
        rng = np.random.default_rng(int(rule_seed))
        Nint = int(round(n.sum()))
        Wbs = []
        for _ in range(BBOOT_):
            nb = rng.multinomial(Nint, n / n.sum()).astype(float)
            fb, _ = fit_npmle_certified(nb, grid, TAU, 1, delta=DELTA, Psi=Psi)
            if fb.n_support > n_atoms or not np.isfinite(fb.loglik):
                continue
            Wbs.append(_trajectory(fb.weights, cache, beta))
        n_boot = len(Wbs)
        ens_secs = time.perf_counter() - t0
        if n_boot:
            Wb = np.vstack(Wbs)
            dec["bagged_npmle"] = (argmax_date(Wb.mean(axis=0)), "", ens_secs)
            d_rc, _ = expected_regret_rule(Wb)
            dec["regret_calibrated"] = (int(d_rc), "" if d_rc > 0 else "invalid_ensemble", 0.0)
        else:
            dec["bagged_npmle"] = (-1, "empty_ensemble", ens_secs)
            dec["regret_calibrated"] = (-1, "empty_ensemble", 0.0)

        def beta_rule():
            o = parametric_beta(n, T, 1, TAU, beta, moments=m_hat)
            return (o.selected_date, o.failure_reason) if o.success else (-1, o.failure_reason)
        run("parametric_beta", beta_rule)

        t0 = time.perf_counter()
        try:
            b = compute_bounds(m_hat, TAU, 1, T, beta, grid_size=LP_GRID, cov=cov,
                               scan=LP_SCAN, set_mode=FROZEN_SET_MODE)
            if not b.feasible:
                raise RuntimeError(b.status)
            secs = time.perf_counter() - t0
            d = decide(b)
            dec["achieved_welfare_lp"] = (d["selected_date"], d.get("failure_reason", ""), secs)
            finite = {int(k): v for k, v in b.lower.items() if np.isfinite(v)}
            best = max(finite.values())
            dec["maximin_welfare"] = (min(k for k, v in finite.items() if v == best), "", 0.0)
            t1 = time.perf_counter()
            S = IdentifiedSet.from_confidence_region(build_region(m_hat, cov, TAU), TAU,
                                                     support_grid(LP_GRID))
            de = decide_envelope(S, b.dates, b.lower, 1, T, beta, scan=ENV_SCAN)
            dec["worst_case_envelope"] = (de["selected_date"], "", time.perf_counter() - t1)
        except Exception as exc:                              # noqa: BLE001
            reason = f"exception:{type(exc).__name__}: {exc}"
            for name in ("achieved_welfare_lp", "maximin_welfare", "worst_case_envelope"):
                dec.setdefault(name, (-1, reason, time.perf_counter() - t0))

        assert set(dec) == set(PROCEDURES)
        for name in PROCEDURES:
            d, reason, secs = dec[name]
            ok = d >= 1
            rows.append({
                "config_id": pair["config_id"], "distribution": pair["distribution"],
                "T": T, "beta": beta, "N": N, "tau": TAU, "seed_index": seed,
                "sim_seed": sim_seed, "procedure": name, "success": ok,
                "failure_reason": reason, "selected_date": d,
                "regret_lower": pair["regret"][0][d] if ok else np.nan,
                "regret_upper": pair["regret"][1][d] if ok else np.nan,
                # the sensitivity check in D.3 assigns date 1 on failure
                "regret_lower_fill1": pair["regret"][0][d if ok else 1],
                "regret_upper_fill1": pair["regret"][1][d if ok else 1],
                "n_bootstrap_used": n_boot if name in ("bagged_npmle", "regret_calibrated") else np.nan,
                "zeta_over_2": pair["zeta_over_2"], "runtime_s": secs,
            })
    tmp = ck.with_suffix(".csv.part")
    pd.DataFrame(rows).to_csv(tmp, index=False)
    tmp.replace(ck)
    ck.with_suffix(".json").write_text(json.dumps({"fingerprint": fp, "rows": len(rows)}) + "\n")
    return str(ck)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Appendix D.3, " + VERSION)
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seeds", default=None, help="comma-separated seed indices")
    ap.add_argument("--ns", default=None, help="comma-separated sample sizes")
    ap.add_argument("--n-boot", type=int, default=None)
    ap.add_argument("--pairs", type=int, default=None, help="use only the first K pairs")
    ap.add_argument("--smoke", action="store_true", help="2 pairs, N=1e3, one seed, 5 bootstrap fits")
    ap.add_argument("--force", action="store_true", help="overwrite an existing raw file")
    ap.add_argument("--discard-stale", action="store_true",
                    help="recompute cells whose checkpoint is stale instead of stopping")
    a = ap.parse_args(argv)
    seeds = [int(x) for x in a.seeds.split(",")] if a.seeds else list(SEEDS)
    ns = [int(x) for x in a.ns.split(",")] if a.ns else list(NS)
    n_boot = a.n_boot if a.n_boot is not None else BBOOT
    n_pairs = a.pairs
    if a.smoke:
        seeds, ns, n_boot, n_pairs = seeds[:1], [1000], 5, 2
    settings = settings_of(seeds, n_boot)
    publication = (seeds == list(SEEDS) and ns == list(NS) and n_boot == BBOOT and n_pairs is None)

    out = Path(a.out).resolve()
    raw_path = out / "matched_info_pairs_raw.csv"
    if raw_path.exists() and not a.force:
        sys.exit(f"{raw_path} exists; pass --force to replace it (or choose another --out)")
    if out == RESULTS.resolve() and not publication:
        sys.exit("refusing to write non-publication settings into results/")
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    pairs = exact_pairs()[:n_pairs]
    print(f"{len(pairs)} exact tau={TAU} pairs reconstructed in "
          f"{time.perf_counter() - t0:.0f}s ({VERSION}; "
          f"{'publication' if publication else 'NON-publication'} settings)", flush=True)
    tasks = [(p, N, settings, str(out / "matched_info_pairs_cells"), a.discard_stale)
             for p in pairs for N in ns]
    with mp.Pool(a.workers) as pool:
        paths = []
        for i, path in enumerate(pool.imap_unordered(run_cell, tasks), 1):
            paths.append(path)
            print(f"[{i}/{len(tasks)}] {Path(path).name} "
                  f"({(time.perf_counter() - t0) / 60:.1f} min)", flush=True)
    df = pd.concat([pd.read_csv(p) for p in sorted(paths)], ignore_index=True)
    df = df.sort_values(["config_id", "N", "seed_index", "procedure"])
    expected = len(pairs) * len(ns) * len(seeds) * len(PROCEDURES)
    if len(df) != expected or df.duplicated(["config_id", "N", "seed_index", "procedure"]).any():
        sys.exit(f"assembled {len(df)} rows, expected {expected} unique")
    tmp = raw_path.with_suffix(".csv.part")
    df.to_csv(tmp, index=False)
    tmp.replace(raw_path)

    from wtm.provenance import write_manifest
    used = df.drop_duplicates(["config_id", "N", "seed_index"])
    boot = df[df.procedure == "bagged_npmle"].n_bootstrap_used
    write_manifest(out / "matched_info_pairs_manifest.json",
                   artifact="matched_info_pairs_raw.csv", artifacts=[raw_path],
                   version=VERSION, publication_settings=publication, settings=settings,
                   sample_sizes=ns, pairs=[p["config_id"] for p in pairs],
                   pair_source="extreme_pair_exact.py, tau=2 exact rational pairs",
                   n_rows=int(len(df)), n_datasets=int(len(used)),
                   bootstrap={"attempted_per_dataset": n_boot,
                              "retained_min": int(boot.min()), "retained_max": int(boot.max()),
                              "retained_total": int(boot.sum())},
                   exclusions="none at generation; the analysis keeps jointly valid trials",
                   failures=df[~df.success.astype(bool)].groupby("procedure").size().to_dict(),
                   solver={"npmle_grid": 200, "npmle_delta": DELTA, "lp_grid": LP_GRID,
                           "lp_scan": LP_SCAN, "envelope_scan": ENV_SCAN, "lp_solver": "HiGHS (scipy)"},
                   core_hours=float(df.runtime_s.sum() / 3600))
    print(f"wrote {raw_path} ({len(df)} rows) in {(time.perf_counter() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
