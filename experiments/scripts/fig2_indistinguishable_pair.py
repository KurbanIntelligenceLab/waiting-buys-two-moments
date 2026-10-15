"""Figure 2 data: two exactly observationally equivalent populations with
different optimal dates.

The figure in the manuscript (paper/FinalVersion-ICLR27/Figures/fig2.png) is the
author's drawing, taken from the 2026-09-30 PDF. This script computes, in exact
arithmetic, every number that drawing prints, and release/verify_paper.py checks
the drawing's annotations against them:
  fig2_records.csv       the five exit-record cell probabilities of panel (a)
                         (exit at t=1 with k=0,1; active at t=2 with k=0,1,2)
  fig2_welfare.csv       the welfare curves of panel (b), optima 5 and 1, and the
                         13.0% regret of s=5 under Q1
It also writes its own rendering, fig2_generated.{png,pdf}, which shows cohort
probabilities instead of exit records and is NOT the manuscript figure.

Uses the exact rational extreme pair of `extreme_pair_exact.py` for the
configuration Beta(1,32), T=20, beta=0.01 at tau=2 (config fad987cc30ad), the
pair with the largest exhibited regret bound (zeta/2 = 6.34%). Q0 is the
low-horizon-survival member (optimal date 5), Q1 the high one (optimal date 1).

(a) prefix observation probabilities of every cohort cell (s, k), s <= tau --
    equal in exact arithmetic, not just to plotting precision;
(b) welfare curves W^s(Q_i) / max_r W^r(Q_i).

Writes the CSVs and fig2_generated.{png,pdf} to --out
(default runs/indistinguishable_pair).
Usage: python experiments/scripts/fig2_indistinguishable_pair.py [--out DIR]
"""
from __future__ import annotations

import sys
from fractions import Fraction
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "runs" / "indistinguishable_pair"

from wtm.config import load_frozen, population_cells  # noqa: E402
from wtm.decision_id import DecisionProblem, eval_mixed_exact  # noqa: E402
from wtm.lp import IdentifiedSet, support_grid  # noqa: E402
from wtm.observation import individual_functionals, phi  # noqa: E402
from experiments.scripts.extreme_pair_exact import (GRID_SIZE, exact_member,  # noqa: E402
                                                    exact_trajectory, extreme_members)

DIST, T, BETA, TAU = "beta_1_32", 20, 0.01, 2
BLUE, ORANGE = "#2a78d6", "#eb6834"   # categorical slots 1-2 (validated pair)
INK, MUTED, GRID = "#1a1a19", "#6b6a63", "#e4e3dc"


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT))
    out = Path(ap.parse_args(argv).out)
    cell = next(c for c in population_cells(load_frozen())
                if c.gamma == 1 and c.dist_name == DIST and c.T == T and c.beta == BETA)
    prob = DecisionProblem(cell.dist, T, BETA, TAU)
    grid = support_grid(GRID_SIZE)
    lo, hi = extreme_members(grid, IdentifiedSet(cell.dist, TAU, grid), T)
    Q = [exact_member(prob, lo.x), exact_member(prob, hi.x)]

    cells, obs = [], [[], []]
    for s in range(1, TAU + 1):
        for k in range(s + 1):
            cells.append(f"s={s}\nk={k}")
            for i, (a, w) in enumerate(Q):
                obs[i].append(sum((wi * eval_mixed_exact(phi(s, k, 1), x)
                                   for x, wi in zip(a, w)), Fraction(0)))
    assert obs[0] == obs[1], "observation laws must agree exactly"

    # exit-record cells of the length-tau window (panel (a) of the manuscript figure)
    records = []
    for key, f in individual_functionals(TAU, 1):
        vals = [sum((wi * eval_mixed_exact(f, x) for x, wi in zip(a, w)), Fraction(0)) for a, w in Q]
        assert vals[0] == vals[1], "record laws must agree exactly"
        records.append({"record": f"{key[0]} t={key[1]} k={key[2]}", "prob_exact": str(vals[0]),
                        "prob": float(vals[0])})

    W = [exact_trajectory(prob, a, w) for a, w in Q]
    dates = prob.dates
    rel = [[float(Wi[s] / max(Wi.values())) for s in dates] for Wi in W]
    opt = [max(dates, key=lambda s: Wi[s]) for Wi in W]
    assert opt == [5, 1], opt

    dest = out
    dest.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(dest / "fig2_records.csv", index=False)
    pd.DataFrame({"cell": [c.replace("\n", " ") for c in cells],
                  "prob_Q0": [float(x) for x in obs[0]],
                  "prob_Q1": [float(x) for x in obs[1]]}).to_csv(dest / "fig2_observations.csv", index=False)
    pd.DataFrame({"date": dates, "W_rel_Q0": rel[0], "W_rel_Q1": rel[1],
                  "W_Q0": [float(W[0][s]) for s in dates],
                  "W_Q1": [float(W[1][s]) for s in dates]}).to_csv(dest / "fig2_welfare.csv", index=False)

    plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
                         "axes.spines.right": False})
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 2.7), gridspec_kw={"width_ratios": [1, 1.35]})

    x = np.arange(len(cells))
    wbar = 0.36
    a.bar(x - wbar / 2 - 0.01, [float(v) for v in obs[0]], wbar, color=BLUE, label=r"$Q_0$")
    a.bar(x + wbar / 2 + 0.01, [float(v) for v in obs[1]], wbar, color=ORANGE,
          hatch="////", edgecolor="white", linewidth=0, label=r"$Q_1$")
    a.set_xticks(x, cells)
    a.set_ylabel("Probability")
    a.set_title("(a) Prefix observation probabilities", loc="left", fontsize=9, color=INK)
    a.grid(axis="y", color=GRID, linewidth=0.6)
    a.set_axisbelow(True)
    a.legend(frameon=False, loc="upper right")

    b.plot(dates, rel[0], color=BLUE, lw=2, marker="o", ms=4, label=r"$Q_0$")
    b.plot(dates, rel[1], color=ORANGE, lw=2, ls="--", marker="s", ms=4, label=r"$Q_1$")
    for i, (c, m) in enumerate(((BLUE, "o"), (ORANGE, "s"))):
        s = opt[i]
        b.plot([s], [1.0], marker=m, ms=9, mfc="white", mec=c, mew=2, zorder=5)
        b.annotate(f"$s^\\star={s}$", (s, 1.0), xytext=(-6, 9) if i else (9, 3), ha="left",
                   textcoords="offset points", color=INK, fontsize=8)
    b.set_xlabel("Allocation date $s$")
    b.set_ylabel(r"$W^s(Q_i)\,/\,\max_r W^r(Q_i)$")
    b.set_xticks([1, 5, 10, 15, 19])
    b.set_ylim(0, 1.14)
    b.set_title("(b) Relative welfare by date", loc="left", fontsize=9, color=INK)
    b.grid(axis="y", color=GRID, linewidth=0.6)
    b.set_axisbelow(True)
    b.legend(frameon=False, loc="lower left")

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(dest / f"fig2_generated.{ext}", dpi=300, bbox_inches="tight")
    r0 = 1 - W[0][1] / W[0][5]
    r1 = 1 - W[1][5] / W[1][1]
    print(f"Q0 optimum {opt[0]}, Q1 optimum {opt[1]}; regret of s=5 under Q1 = {float(r1):.4f}, "
          f"of s=1 under Q0 = {float(r0):.4f}")


if __name__ == "__main__":
    main()
