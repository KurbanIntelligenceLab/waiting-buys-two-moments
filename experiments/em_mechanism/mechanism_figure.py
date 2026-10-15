"""Figure 5: the mechanism (results/em_mechanism/figures/fig5_mechanism.{pdf,png})."""
from __future__ import annotations

import numpy as np
import pandas as pd

from experiments.em_mechanism.figures import INIT_COLORS, INK, MUTED, plt, save
from experiments.em_mechanism.mechanism import BIN_LABELS, MOUT


def main():
    fig, axes = plt.subplots(1, 4, figsize=(7.4, 2.5))
    # (a) asymmetric loss in the m3 error
    g = pd.read_csv(MOUT / "regret_by_m3_error.csv")
    ax = axes[0]
    order = list(dict.fromkeys(g.sort_values("m3_err_bin", key=lambda s: s.str.extract(r"\((-?[\d.]+)")[0].astype(float)).m3_err_bin))
    pos = {b: i for i, b in enumerate(order)}
    for N, c in ((1000, "#2a78d6"), (100000, "#eb6834")):
        x = g[g.N == N]
        ax.plot([pos[b] for b in x.m3_err_bin], x.mean_regret, marker="o", ms=3, color=c, label=f"N={N:,}")
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([s.replace("(", "").replace("]", "").replace(", ", " to ") for s in order], rotation=60, fontsize=5)
    ax.axvline(pos[[b for b in order if b.startswith("(-0.1")][0]] + 0.5, color=MUTED, lw=0.7)
    ax.set_title("(a) regret vs error in m3\n(all estimators pooled)")
    ax.set_xlabel("relative error of fitted m3"); ax.set_ylabel("mean relative regret (P)")
    ax.legend(frameon=False)
    # (b) m3 bias by estimator
    b = pd.read_csv(MOUT / "m3_bias_by_estimator.csv")
    ax = axes[1]
    for j, (N, c) in enumerate(((1000, "#2a78d6"), (100000, "#eb6834"))):
        x = b[b.N == N].set_index("estimator").loc[["EM55", "EM_cap", "NPMLE"]]
        ax.bar(np.arange(3) + (j - 0.5) * 0.38, x.share_m3_under, width=0.36, color=c, label=f"N={N:,}")
    ax.axhline(0.5, color=MUTED, lw=0.7, ls="--")
    ax.set_xticks(range(3)); ax.set_xticklabels(["EM55", "EM cap", "NPMLE"])
    ax.set_title("(b) share of datasets with\nm3 under-estimated"); ax.set_ylim(0, 0.75)
    ax.legend(frameon=False, fontsize=5.5, loc="upper left", ncol=2)
    # (c) interventions on worse events
    t = pd.read_csv(MOUT / "constraint_tests.csv.gz")
    ph = pd.read_csv(MOUT / "posthoc_moment_tests.csv.gz")
    ax = axes[2]
    specs = [("T1_minimal_change", t, "d55", "late fit,\nnear EM55"),
             ("T3_freeze_decision_bins", t, "d55", "freeze\ndecision bins"),
             ("T4_freeze_irrelevant_bins", t, "d55", "freeze\nother bins"),
             ("Tno3_all_but_m3", ph, "d55", "late m1,m2;\nEM55 m3*"),
             ("Ttrue3_late_with_true_m3", ph, "d55", "late fit,\ntrue m3*")]
    for j, (N, c) in enumerate(((1000, "#2a78d6"), (100000, "#eb6834"))):
        vals = []
        for name, df, tgt, _ in specs:
            x = df[(df.N == N) & (df.cls == "worse") & df[f"{name}_feasible"].fillna(False).astype(bool)]
            vals.append((x[f"{name}_date"] == x[tgt]).mean() if len(x) else np.nan)
        ax.bar(np.arange(len(specs)) + (j - 0.5) * 0.38, vals, width=0.36, color=c)
    ax.set_xticks(range(len(specs))); ax.set_xticklabels([s[3] for s in specs], fontsize=5, rotation=60)
    ax.set_ylim(0, 1); ax.set_title("(c) worse events: share keeping\nEM55's date (* post hoc)")
    # (d) mean mass shift 55 -> cap
    e = pd.read_csv(MOUT / "events.csv.gz")
    ax = axes[3]
    for j, (cls, c) in enumerate((("worse", INK), ("improve", "#1baf7a"))):
        x = e[(e.N == 1000) & (e.cls == cls)]
        d = [(x[f"wcap_mass{k}"] - x[f"w55_mass{k}"]).mean() for k in range(len(BIN_LABELS))]
        ax.bar(np.arange(len(BIN_LABELS)) + (j - 0.5) * 0.38, d, width=0.36, color=c, label=cls)
    ax.axhline(0, color=MUTED, lw=0.7)
    ax.set_xticks(range(len(BIN_LABELS))); ax.set_xticklabels(BIN_LABELS, rotation=60, fontsize=5)
    ax.set_title("(d) mass shift EM55 -> cap\n(N = 1e3)"); ax.set_ylim(-0.12, 0.16)
    ax.legend(frameon=False, fontsize=5.5, loc="upper center", ncol=2)
    fig.tight_layout()
    save(fig, "fig5_mechanism")


if __name__ == "__main__":
    main()
