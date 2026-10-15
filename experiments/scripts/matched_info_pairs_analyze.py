"""Tables 6 and 7 (Appendix D.3) from matched_info_pairs_raw.csv (reconstruction-v1).

Retention: a trial is kept only if all ten procedures return a decision on it
(joint validity). For pair j and sample size N, with S_{j,N} its retained
trials and R_{j,i} the relative regret under member i,

    rho_{j,N}^a = max_i  mean_{s in S_{j,N}} R_{j,i}(d^a_{j,N,s})
    rho_N^a     = mean_j rho_{j,N}^a                  (Table 6)

Table 7 reports rho_N^a - rho_N^{NPMLE}, with a 95% percentile interval from
2,000 paired bootstrap replicates (seed 20260906). The replicates resample
retained trials within each pair, jointly across procedures, and recompute
the maximum before pooling. The intervals are exploratory and unadjusted for
multiplicity.

The raw file is checked before anything is computed: unique
(pair, N, seed, procedure) keys, exactly the expected 30 pairs x 2 N x 5
seeds x 10 procedures, and finite regrets wherever a decision exists.

Usage:  python experiments/scripts/matched_info_pairs_analyze.py [--raw FILE] [--out DIR]
Writes  <out>/matched_info_pairs_table6.csv, <out>/matched_info_pairs_table7.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
RESULTS = ROOT / "results"

from experiments.scripts.matched_info_pairs import NS, PROCEDURES, SEEDS  # noqa: E402

N_BOOT = 2000
BOOT_SEED = 20260906
TABLE7 = ("em_55", "bagged_npmle", "regret_calibrated")
KEY = ["config_id", "N", "seed_index"]
LABELS = {"certified_npmle": "Certified NPMLE", "em_30": "EM, 30 iterations",
          "em_55": "EM, 55 iterations", "em_80": "EM, 80 iterations",
          "bagged_npmle": "Bagged NPMLE", "regret_calibrated": "Regret calibration",
          "parametric_beta": "Moment-matched Beta", "achieved_welfare_lp": "Achieved-welfare LP",
          "worst_case_envelope": "Worst-case envelope", "maximin_welfare": "Maximin welfare"}


class RawError(ValueError):
    pass


def check_raw(raw: pd.DataFrame, n_pairs=30) -> None:
    dup = raw.duplicated(KEY + ["procedure"])
    if dup.any():
        raise RawError(f"{int(dup.sum())} duplicate (pair, N, seed, procedure) rows")
    if set(raw.procedure) != set(PROCEDURES):
        raise RawError(f"procedures {sorted(set(raw.procedure) ^ set(PROCEDURES))} missing or unexpected")
    if set(raw.N) != set(NS) or set(raw.seed_index) != set(SEEDS) or raw.config_id.nunique() != n_pairs:
        raise RawError("unexpected pairs, sample sizes or seeds")
    per = raw.groupby(KEY).procedure.nunique()
    if len(per) != n_pairs * len(NS) * len(SEEDS) or (per != len(PROCEDURES)).any():
        raise RawError("a dataset lacks some procedures")
    ok = raw.success.astype(str) == "True"
    for c in ("regret_lower", "regret_upper"):
        if not np.isfinite(raw.loc[ok, c]).all():
            raise RawError(f"non-finite {c} on a successful decision")
    for c in ("regret_lower_fill1", "regret_upper_fill1"):
        if not np.isfinite(raw[c]).all():
            raise RawError(f"non-finite {c}")


def pair_risk(df, cols=("regret_lower", "regret_upper")):
    """rho^a_N: worst member of the trial-averaged regret, pooled over pairs."""
    per = df.groupby(["procedure", "N", "config_id"])[list(cols)].mean()
    return per.max(axis=1).groupby(["procedure", "N"]).mean().unstack("N")


def pair_tables(raw: pd.DataFrame, n_boot=N_BOOT, seed=BOOT_SEED) -> dict:
    check_raw(raw)
    ok = raw.assign(success=raw.success.astype(str) == "True")
    valid = ok.groupby(KEY).success.all().rename("jointly_valid").reset_index()
    kept = ok.merge(valid, on=KEY, validate="many_to_one")
    kept = kept[kept.jointly_valid]
    fails = ok[~ok.success]
    key_of = lambda f: set(map(tuple, f[KEY].itertuples(index=False, name=None)))  # noqa: E731
    bad_npmle = key_of(fails[fails.procedure == "certified_npmle"])
    bad_beta = key_of(fails[fails.procedure == "parametric_beta"])
    others = set(fails.procedure) - {"certified_npmle", "parametric_beta"}
    per_pair = {N: valid[(valid.N == N) & valid.jointly_valid].groupby("config_id").size()
                for N in NS}
    if any(len(p) != 30 for p in per_pair.values()):
        raise RawError("a pair has no jointly valid trial")

    t6 = pair_risk(kept).loc[list(PROCEDURES)]
    filled = pair_risk(ok, ("regret_lower_fill1", "regret_upper_fill1")).loc[list(PROCEDURES)]

    rng = np.random.default_rng(seed)
    t7 = []
    for N, g in kept.groupby("N"):
        pairs = sorted(g.config_id.unique())
        cube = {}
        for j in pairs:
            gj = g[g.config_id == j].pivot(index="seed_index", columns="procedure",
                                           values=["regret_lower", "regret_upper"])
            cube[j] = np.stack([gj["regret_lower"][list(PROCEDURES)].values.T,
                                gj["regret_upper"][list(PROCEDURES)].values.T], axis=-1)

        def pooled(idx):
            return np.mean([cube[j][:, idx[j], :].mean(axis=1).max(axis=1) for j in pairs], axis=0)

        point = pooled({j: np.arange(cube[j].shape[1]) for j in pairs})
        boots = np.array([pooled({j: rng.integers(0, cube[j].shape[1], cube[j].shape[1])
                                  for j in pairs}) for _ in range(n_boot)])
        ref = PROCEDURES.index("certified_npmle")
        for name in TABLE7:
            a = PROCEDURES.index(name)
            diff = boots[:, a] - boots[:, ref]
            t7.append({"procedure": name, "N": int(N), "difference": point[a] - point[ref],
                       "ci_lo": np.percentile(diff, 2.5), "ci_hi": np.percentile(diff, 97.5)})
    t7 = pd.DataFrame(t7)

    f4 = lambda x: f"{x:.4f}"  # noqa: E731
    vals = {
        "ret_1e3": str(int(per_pair[1000].sum())), "ret_1e5": str(int(per_pair[100000].sum())),
        "pp_min": str(int(min(p.min() for p in per_pair.values()))),
        "pp_max": str(int(max(p.max() for p in per_pair.values()))),
        "npmle_fail": str(len(bad_npmle)), "beta_fail": str(len(bad_beta - bad_npmle)),
        "fallback": f4((filled - t6).abs().values.max()),
        "table6": "; ".join(f"{LABELS[p]} {f4(t6.loc[p, 1000])}/{f4(t6.loc[p, 100000])}"
                            for p in PROCEDURES),
        "table7": "; ".join(f"{LABELS[r.procedure]} N={r.N} {r.difference:+.8f} "
                            f"[{r.ci_lo:+.8f}, {r.ci_hi:+.8f}]" for r in t7.itertuples()),
    }
    em = t7[t7.procedure == "em_55"].set_index("N")
    excl = [N for N in NS if em.loc[N, "ci_hi"] < 0 or em.loc[N, "ci_lo"] > 0]
    vals["interp"] = ("EM55 difference negative at both N" if (em.difference < 0).all()
                      else "EM55 difference not negative at both N") + \
        f"; interval excludes zero at N in {excl}"
    return {"table6": t6, "table7": t7, "values": vals, "others_fail": sorted(others),
            "kept": kept}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=str(RESULTS / "matched_info_pairs_raw.csv"))
    ap.add_argument("--out", default=str(RESULTS))
    a = ap.parse_args()
    raw = pd.read_csv(a.raw)
    r = pair_tables(raw)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    r["table6"].to_csv(out / "matched_info_pairs_table6.csv")
    r["table7"].to_csv(out / "matched_info_pairs_table7.csv", index=False)
    v = r["values"]
    print(f"retained {v['ret_1e3']}/150 (N=1e3), {v['ret_1e5']}/150 (N=1e5); "
          f"{v['pp_min']}-{v['pp_max']} trials per pair; NPMLE invalid {v['npmle_fail']}, "
          f"Beta-only failures {v['beta_fail']}; other procedures failing: {r['others_fail'] or 'none'}")
    print(f"date 1 on failure: max change {v['fallback']}")
    print("\nTable 6 (reconstruction-v1)\n" + r["table6"].round(4).to_string())
    print("\nTable 7 (reconstruction-v1)")
    for x in r["table7"].itertuples():
        print(f"  {x.procedure:18s} N={x.N:<6d} {x.difference:+.8f} [{x.ci_lo:+.8f}, {x.ci_hi:+.8f}]")
    print("\n" + v["interp"])


if __name__ == "__main__":
    main()
