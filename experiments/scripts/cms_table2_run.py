"""CMS on exactly the sample underlying the manuscript's Table 3.

TWO REQUIRED FLAGS. The archived Table 3 CMS row is reproduced, draw for draw,
only by `--resampling paths --rng legacy` (results/cms_table2_cells/,
results/cms_table2_raw.csv).

  --resampling paths | prefailure
      paths resamples the raw path cells, which separate period-tau exits
      from survivors; prefailure merges those cells first. For the idealized
      procedure (bootstrap randomness independent of the data) the two give
      the recommendation the same distribution; they are never seed-identical.
  --rng legacy | independent
      legacy uses the archived seeds, draw d -> base + seed + d, which reuse
      the run's own data seed (draw 0) and other runs' data and bootstrap
      seeds. independent uses per-draw SeedSequence children of the frozen
      protocol's rule seed (wtm.cms.draw_seed), as the protocol requires.

`--resampling paths --rng legacy` is the manuscript's implementation. The other
combinations are alternatives proposed for the authors' decision, not
replacements. Each writes to its own checkpoint directory (ckpt_dir), so
nothing here can overwrite or mix with the archive.
See paper/reconciliation/README.md section 06.

THE SAMPLE. Those tables are tau = 2 only, reported at N = 10^3 and N = 10^5,
pooled over all 30 population configurations and all 20 seeds:
30 x 2 x 20 = 1,200 runs. Fifteen of the thirty configurations are at T = 20,
so a T = 12-only result cannot be inserted into these columns.

ORDERING. Tasks run seed-major: every configuration at seed 1001 finishes
before seed 1002 begins. Each seed therefore completes a coherent, balanced
version of the whole table -- all 30 configurations, both sample sizes, both
horizons -- so the run can be stopped at any seed boundary and what exists is a
matched subset rather than a fragment. Within a seed, T = 12 runs first because
they cost 2.5x less.

CHECKPOINTS carry the seed in the filename. The earlier full-grid runner fixed
the seed at 1001 and left it out of the name; reusing that convention here
would make seed 1002 silently overwrite seed 1001.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
RESULTS = ROOT / "results"
CKPT = RESULTS / "cms_table2_cells"     # --resampling paths --rng legacy (the archive)


def ckpt_dir(resampling: str, rng: str) -> Path:
    """The archive lives only under (paths, legacy); every other combination
    writes elsewhere, so a new run can never overwrite or mix with it."""
    if (resampling, rng) == ("paths", "legacy"):
        return CKPT
    if (resampling, rng) == ("prefailure", "legacy"):
        return RESULTS / "cms_table2_prefailure_cells"
    return RESULTS / f"cms_table2_{resampling}_{rng}_cells"
LEGACY = RESULTS / "cms_full_cells"          # seed-1001 draws already computed

N_BOOT = 20
TAU = 2
NS = [1000, 100000]
SEEDS = list(range(1001, 1021))
GRID = 401


def _tag(dist_name, T, beta, N, seed, draw):
    return f"{dist_name}_{T}_{beta}_{N}_t{TAU}_s{seed}_d{draw:03d}.json"


def adopt_legacy() -> int:
    """Copy the seed-1001 draws already computed under the full-grid runner."""
    if not LEGACY.exists():
        return 0
    CKPT.mkdir(exist_ok=True)
    n = 0
    for f in LEGACY.glob("*.json"):
        for r in json.loads(f.read_text()):
            if r["prefix_t"] != TAU or r["N"] not in NS:
                continue
            dst = CKPT / _tag(r["distribution"], r["T"], r["beta"], r["N"],
                              r["seed_index"], r["draw"])
            if not dst.exists():
                dst.write_text(json.dumps([r]))
                n += 1
    return n


def run_draw(args):
    dist_name, T, beta, N, seed, draw, resampling, rng = args
    ckpt = ckpt_dir(resampling, rng)
    ckpt.mkdir(exist_ok=True)
    ck = ckpt / _tag(dist_name, T, beta, N, seed, draw)
    if ck.exists():
        return str(ck)

    from wtm import cms
    from wtm.config import config_id, load_frozen, population_cells
    from wtm.moment_projection import empirical_path_counts
    from wtm.simulate import simulate_prefix
    from wtm.welfare import PopulationConfig

    cfg = load_frozen()
    cell = next(c for c in population_cells(cfg)
                if c.gamma == 1 and c.dist_name == dist_name
                and c.T == T and c.beta == beta)
    pop = PopulationConfig(cell.dist, 1, T, beta)
    cell_id = config_id(dist=dist_name, gamma=1, T=T, beta=beta, N=N, t=TAU)
    base = int(cell_id, 16)

    rec = {"rule": "cms_bootstrap", "cell_id": cell_id, "distribution": dist_name,
           "T": T, "beta": beta, "N": N, "prefix_t": TAU, "seed_index": seed,
           "draw": draw, "true_optimal_date": pop.t_star, "resampling": resampling,
           "rng": rng}
    try:
        data = simulate_prefix(cell.dist, N, T, 1, TAU,
                               seed=(base + seed) % 2 ** 32)
        counts = np.asarray(empirical_path_counts(data, TAU), dtype=float)
        res = cms.bootstrap_rule(counts, TAU, 1, T, beta, n_boot=1,
                                 grid_size=GRID, resampling=resampling,
                                 seed=cms.draw_seed(rng, base, seed, draw))
        if not res.per_draw:
            rec.update(success=False,
                       failure_reason=res.status or "draw unusable",
                       n_projected=res.n_projected,
                       n_infeasible=res.n_infeasible, runtime_s=res.runtime_s)
        else:
            rec.update(success=True, failure_reason="",
                       max_risk=json.dumps({str(k): v for k, v
                                            in res.per_draw[0].items()}),
                       n_projected=res.n_projected,
                       n_infeasible=res.n_infeasible, n_exact=res.n_exact,
                       n_lp_solved=res.n_lp_solved, runtime_s=res.runtime_s)
    except Exception as exc:                                       # noqa: BLE001
        rec.update(success=False, failure_reason=f"{type(exc).__name__}: {exc}")
    ck.write_text(json.dumps([rec]))
    return str(ck)


def main() -> None:
    from wtm.config import load_frozen, population_cells
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--seeds", type=int, default=len(SEEDS),
                    help="how many seeds to target, seed-major from 1001")
    ap.add_argument("--seed-from", type=int, default=SEEDS[0],
                    help="first seed this machine owns (inclusive)")
    ap.add_argument("--seed-to", type=int, default=SEEDS[-1],
                    help="last seed this machine owns (inclusive). Two machines "
                         "given disjoint ranges never write the same checkpoint, "
                         "so they can share the work through git with no locking.")
    ap.add_argument("--resampling", choices=["paths", "prefailure"], required=True,
                    help="paths: reproduces the archived Table 3 draws seed for seed; "
                         "prefailure: merges period-tau exit cells first (same "
                         "distribution, different draws for a given seed)")
    ap.add_argument("--rng", choices=["legacy", "independent"], required=True,
                    help="legacy: the archived seeds (draw d uses base + seed + d, "
                         "which reuses data seeds; reproduction only); independent: "
                         "per-draw SeedSequence children of the protocol's rule seed")
    a = ap.parse_args()

    adopted = adopt_legacy() if (a.resampling, a.rng) == ("paths", "legacy") else 0
    ckpt = ckpt_dir(a.resampling, a.rng)
    cfg = load_frozen()
    cells = [c for c in population_cells(cfg) if c.gamma == 1]
    seeds = [s for s in SEEDS[:a.seeds]
             if a.seed_from <= s <= a.seed_to]
    tasks = [(c.dist_name, c.T, c.beta, N, s, d, a.resampling, a.rng)
             for s in seeds for c in cells for N in NS
             for d in range(a.n_boot)]
    tasks.sort(key=lambda x: (seeds.index(x[4]), x[1], x[0], x[3], x[5]))
    todo = [t for t in tasks if not (ckpt / _tag(*t[:6])).exists()]
    print(f"adopted {adopted} existing draws; target {len(tasks)} draws "
          f"({len(tasks)//a.n_boot} runs over {len(seeds)} seeds); "
          f"{len(todo)} remaining", flush=True)
    if not todo:
        return
    t0 = time.perf_counter()
    k = 0
    with mp.Pool(a.workers) as pool:
        # imap_unordered, not imap: ordered delivery yields results only in
        # submission order, so the counter tracks the longest completed PREFIX
        # rather than throughput. With one slow task at the head, fifteen
        # workers can finish and the rate still reads as though nothing has.
        for _ in pool.imap_unordered(run_draw, todo, chunksize=1):
            k += 1
            if k % 25 == 0 or k == len(todo):
                el = time.perf_counter() - t0
                print(f"  [{k}/{len(todo)}] {el/60:.1f} min, {el/k:.1f} s/draw",
                      flush=True)


if __name__ == "__main__":
    main()
