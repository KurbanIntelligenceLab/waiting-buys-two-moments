"""Exact accounting for the witness construction of Appendix B.2.

Each stored witness is Q = (1 - eps) A + eps B, dB = (1 + g) dP, with A a
rational atomic measure, P the reference population, and g a polynomial of
degree <= n = 2 tau - 1. For every stored witness this computes, in exact
rational arithmetic:

  mass(A) = sum w_i            (not required to equal 1)
  E_P[g]                        = -((1 - eps) / eps) (mass(A) - 1), predicted
  mass(B) = 1 + E_P[g]
  mass(Q) = (1 - eps) mass(A) + eps mass(B)          (must equal 1)
  max_j |E_Q[p^j] - E_P[p^j]|, j = 0..n               (must be 0)
  min_i w_i >= 0, eps in (0,1), the Bernstein lower bound of 1 + g on [0, 1] (>= 0)

    python experiments/audit/witness_mass_audit.py [--out DIR]
"""
import argparse
import gzip
import json
import sys
from fractions import Fraction
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.scripts.verify_decision_certificates import cells_gamma1, parse  # noqa: E402
from wtm.decision_id import bernstein_lower_bound, mono_to_bernstein  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "results" / "audit"))
    a = ap.parse_args()
    cells = cells_gamma1()
    rows = []
    for run in ("full", "actionable"):
        for f in sorted((ROOT / "results/decision_identification/raw" / run).glob("*.json.gz")):
            job = json.load(gzip.open(f, "rt"))
            dist = cells[job["config_id"]].dist
            n = 2 * job["tau"] - 1
            mu = [dist.mixed_exact(j, 0) for j in range(2 * n + 2)]
            for p in job["pairs"]:
                if p["status"] != "witness":
                    continue
                w = p["witness"]
                atoms = [parse(x) for x in w["atoms"]]
                wts = [parse(x) for x in w["weights"]]
                eps = parse(w["eps"])
                g = [parse(x) for x in w["g"]]
                massA = sum(wts, Fraction(0))
                EPg = sum((c * mu[i] for i, c in enumerate(g)), Fraction(0))
                massB = 1 + EPg
                massQ = (1 - eps) * massA + eps * massB
                resid = []
                for j in range(n + 1):
                    EA = sum((x * t ** j for t, x in zip(atoms, wts)), Fraction(0))
                    EB = mu[j] + sum((c * mu[j + i] for i, c in enumerate(g)), Fraction(0))
                    resid.append(abs((1 - eps) * EA + eps * EB - mu[j]))
                pred = -((1 - eps) / eps) * (massA - 1)
                rows.append({
                    "run": run, "config_id": job["config_id"], "tau": job["tau"],
                    "d": p["d"], "r": p["r"], "n_atoms": len(atoms), "deg_g": len(g) - 1, "n": n,
                    "eps": float(eps), "massA_minus_1": float(massA - 1),
                    "E_P_g": float(EPg), "E_P_g_equals_prediction": EPg == pred,
                    "massB_minus_1": float(massB - 1), "massQ_exactly_1": massQ == 1,
                    "max_moment_residual": float(max(resid)), "moments_exact": max(resid) == 0,
                    "min_weight_nonneg": min(wts) >= 0, "eps_in_0_1": 0 < eps < 1,
                    "one_plus_g_bernstein_lb": float(bernstein_lower_bound(mono_to_bernstein([1 + g[0]] + g[1:]))),
                    "deg_g_le_n": len(g) - 1 <= n,
                })
    df = pd.DataFrame(rows)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "witness_mass_audit.csv", index=False)
    s = {"witnesses": len(df),
         "massA_ne_1": int((df.massA_minus_1 != 0).sum()),
         "max_abs_massA_minus_1": float(df.massA_minus_1.abs().max()),
         "max_abs_massB_minus_1": float(df.massB_minus_1.abs().max()),
         "all_E_P_g_equal_prediction": bool(df.E_P_g_equals_prediction.all()),
         "all_massQ_exactly_1": bool(df.massQ_exactly_1.all()),
         "all_moments_exact": bool(df.moments_exact.all()),
         "all_weights_nonneg": bool(df.min_weight_nonneg.all()),
         "all_eps_in_0_1": bool(df.eps_in_0_1.all()),
         "min_bernstein_lb_1_plus_g": float(df.one_plus_g_bernstein_lb.min()),
         "all_deg_g_le_n": bool(df.deg_g_le_n.all()),
         "eps_values": sorted(df.eps.unique().tolist())}
    (out / "witness_mass_audit.json").write_text(json.dumps(s, indent=2) + "\n")
    print(json.dumps(s, indent=2))


if __name__ == "__main__":
    main()
