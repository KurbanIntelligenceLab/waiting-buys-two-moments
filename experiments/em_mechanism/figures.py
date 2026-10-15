"""Figures for the EM-mechanism study (PDF + PNG) from results/em_mechanism/analysis/.

Fig 1  optimization versus decision quality (exact probabilities; finite samples)
Fig 2  initialization sensitivity (aggregate and per-configuration)
Fig 3  compatible populations (one trajectory under both members; welfare curves;
       pooled worst-member risk; position inside the identified interval)
Fig 4  grid resolution and observation length

Representative configurations are fixed by rule in analyze.py (REP_T, REP_BETA;
PAIR_*), not chosen from results.

Usage: python -m experiments.em_mechanism.figures [--finite finite] [--population population]
"""
from __future__ import annotations

import argparse

import matplotlib
import matplotlib.ticker
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from experiments.em_mechanism.analyze import (AN, PAIR_BETA, PAIR_DIST, PAIR_T, REP_BETA,  # noqa: E402
                                              REP_T)
from experiments.em_mechanism.common import OUT  # noqa: E402

FIG = OUT / "figures"
# Reference categorical palette, fixed slot order (one slot per initialization).
INIT_COLORS = {"uniform": "#2a78d6", "low": "#eb6834", "middle": "#1baf7a", "high": "#eda100",
               "rand1": "#e87ba4", "rand2": "#008300", "rand3": "#4a3aa7", "rand4": "#e34948"}
TRUTH_STYLE = {"ref": ("#2a78d6", "-", "reference P"), "lower": ("#eb6834", "--", "lower member"),
               "upper": ("#1baf7a", ":", "upper member")}
INK, MUTED = "#0b0b0b", "#52514e"
FAM_LABEL = {"beta_1_32": "Beta(1,32)", "beta_1_8": "Beta(1,8)", "beta_1_4": "Beta(1,4)",
             "mixture_high_risk_subgroup": "0.95B(1,32)+0.05B(8,2)",
             "bimodal": "0.5B(3,57)+0.5B(12,28)"}
FAMS = list(FAM_LABEL)

plt.rcParams.update({"font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5,
                     "legend.fontsize": 6.5, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "lines.linewidth": 1.4, "axes.grid": True,
                     "grid.color": "#e6e5e1", "grid.linewidth": 0.5, "pdf.fonttype": 42})


def xsym(ax, cap=None):
    ax.set_xscale("symlog", linthresh=10, linscale=0.6)
    ax.set_xlim(0, cap)
    if cap is not None:
        ticks = [0, 10] + [10 ** e for e in (3, 5, 7) if 10 ** e <= cap * 1.01]
        ax.set_xticks(ticks)
        ax.set_xticklabels(["0", "10"] + [f"1e{e}" for e in (3, 5, 7) if 10 ** e <= cap * 1.01])
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    for k in (30, 55, 80):
        ax.axvline(k, color="#c3c2b7", lw=0.6, zorder=0)


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIG / f"{name}.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def read(name, **kw):
    p = AN / name
    return pd.read_csv(p, **kw) if p.exists() else None


# ------------------------------------------------------------------ Fig 1
def fig1(pop_sub, finite_label):
    dec = read("population_by_config_key_iters.csv.gz")
    pdir = OUT / pop_sub
    fits = pd.read_csv(pdir / "tau2_G200_continuous_fits.csv.gz")
    decf = pd.read_csv(pdir / "tau2_G200_continuous_decisions.csv.gz")
    truth = pd.read_csv(pdir / "tau2_G200_continuous_truth.csv.gz")
    cap = int(fits.iteration.max())
    fig, axes = plt.subplots(3, 5, figsize=(7.2, 4.6), sharex=True)
    for j, fam in enumerate(FAMS):
        f = fits[(fits.distribution == fam) & (fits.init == "uniform")]
        npf = fits[(fits.distribution == fam) & (fits.init == "npmle")].iloc[0]
        d = decf[(decf.distribution == fam) & (decf["T"] == REP_T) & (decf.beta == REP_BETA)]
        de = d[d.init == "uniform"].sort_values("iteration")
        dn = d[d.init == "npmle"].iloc[0]
        ax = axes[0, j]
        ax.plot(f.iteration, np.maximum(f.gap, 1e-17), color=INK, label="likelihood gap")
        ax.plot(f.iteration, np.maximum(f.kkt, 1e-17), color=MUTED, ls="--", lw=1, label="KKT residual")
        ax.set_yscale("log"); ax.set_ylim(1e-14, 10)
        ax.axhline(1e-8, color="#c3c2b7", lw=0.6)
        ax.set_title(FAM_LABEL[fam], fontsize=7)
        ax = axes[1, j]
        for k_, (c, ls, lab) in TRUTH_STYLE.items():
            ax.plot(de.iteration, de[f"regret_{k_}"], color=c, ls=ls, label=lab)
            ax.plot([cap * 1.6], [dn[f"regret_{k_}"]], marker="D", ms=4, color=c, ls="none", clip_on=False)
        ax = axes[2, j]
        ax.step(de.iteration, de.date, where="post", color=INK)
        tr = truth[(truth.distribution == fam) & (truth["T"] == REP_T) & (truth.beta == REP_BETA)]
        for k_, (c, ls, lab) in TRUTH_STYLE.items():
            tt = tr[tr.truth == k_]
            ax.axhline(int(tt.s[tt.regret == 0].min()), color=c, ls=ls, lw=0.9)
        ax.plot([cap * 1.6], [dn.date], marker="D", ms=4, color=INK, ls="none", clip_on=False)
        for r in range(3):
            xsym(axes[r, j], cap * 1.7)
        axes[2, j].set_xlabel("EM iteration k")
    axes[0, 0].set_ylabel("gap to grid optimum\n(per individual)")
    axes[1, 0].set_ylabel("relative regret")
    axes[2, 0].set_ylabel("selected date")
    h0, l0 = axes[0, 0].get_legend_handles_labels()
    h1, l1 = axes[1, 0].get_legend_handles_labels()
    fig.legend(h0 + h1, l0 + l1, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle(f"Exact observation probabilities, tau = 2, G = 200, uniform start; T = {REP_T}, "
                 f"beta = {REP_BETA}.  Diamonds at right: certified NPMLE.  Horizontal lines (bottom): "
                 "optimal date of each compatible population.", fontsize=6.5, y=1.0)
    fig.tight_layout()
    save(fig, "fig1a_exact_trajectories")

    path = read(f"{finite_label}_path_by_config.csv.gz")
    summ = read(f"{finite_label}_summary_by_init.csv")
    if path is None:
        return
    p = path[(path.kind == "ref") & (path.init == "uniform")]
    g = p.groupby(["N", "iteration"]).agg(msr=("msr_ref", "mean"), gap=("mean_gap", "mean"),
                                          same=("share_same_date_as_npmle", "mean"),
                                          cert=("share_kkt_le_1e8", "mean")).reset_index()
    Ns = sorted(g.N.unique())
    fig, axes = plt.subplots(3, len(Ns), figsize=(4.8, 4.4), sharex=True, squeeze=False)
    for j, N in enumerate(Ns):
        x = g[g.N == N]
        capf = int(x.iteration.max())
        npm = summ[(summ.kind == "ref") & (summ.N == N) & (summ.procedure == "NPMLE")].MSR_x1e3.iloc[0]
        axes[0, j].plot(x.iteration, x.gap, color=INK)
        axes[0, j].set_yscale("log")
        axes[0, j].set_title(f"N = {N:,}")
        axes[1, j].plot(x.iteration, 1e3 * x.msr, color=INIT_COLORS["uniform"], label="EM (uniform start)")
        axes[1, j].axhline(npm, color=INK, ls="--", lw=1, label="certified NPMLE")
        axes[2, j].plot(x.iteration, x.same, color=INK, label="date = NPMLE date")
        axes[2, j].plot(x.iteration, x.cert, color=MUTED, ls="--", lw=1, label="KKT <= 1e-8")
        axes[2, j].set_ylim(-0.02, 1.02)
        for r in range(3):
            xsym(axes[r, j], capf)
        axes[2, j].set_xlabel("EM iteration k")
    axes[0, 0].set_ylabel("mean likelihood gap")
    axes[1, 0].set_ylabel("MSR x 1e3 (reference P)")
    axes[2, 0].set_ylabel("share of datasets")
    axes[1, 0].legend(frameon=False); axes[2, 0].legend(frameon=False)
    fig.suptitle("Finite samples, 30 configurations x 50 datasets, tau = 2", fontsize=7)
    fig.tight_layout()
    save(fig, "fig1b_finite_trajectories")


# ------------------------------------------------------------------ Fig 2
def fig2(pop_sub, finite_label):
    pdir = OUT / pop_sub
    dec = pd.read_csv(pdir / "tau2_G200_continuous_decisions.csv.gz")
    path = read(f"{finite_label}_path_by_config.csv.gz")
    per = read(f"{finite_label}_by_config.csv")
    summ = read(f"{finite_label}_summary_by_init.csv")
    fig = plt.figure(figsize=(7.2, 4.8))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1])
    ax = fig.add_subplot(gs[0, 0])
    em = dec[dec.iteration >= 0]
    m = em.groupby(["init", "iteration"]).regret_ref.mean().reset_index()
    for init, c in INIT_COLORS.items():
        x = m[m.init == init]
        ax.plot(x.iteration, x.regret_ref, color=c, lw=1.1, label=init)
    npm = dec[dec.init == "npmle"].regret_ref.mean()
    ax.axhline(npm, color=INK, ls="--", lw=1, label="NPMLE")
    xsym(ax, int(em.iteration.max())); ax.set_yscale("symlog", linthresh=1e-4); ax.set_ylim(bottom=0)
    ax.set_title("(a) exact probabilities: mean regret\n(30 configurations)")
    ax.set_xlabel("EM iteration k"); ax.set_ylabel("mean relative regret (P)")
    ax.legend(frameon=False, ncol=2, fontsize=5.5)
    if path is not None:
        for j, N in enumerate(sorted(path.N.unique())):
            ax = fig.add_subplot(gs[0, j + 1])
            p = path[(path.kind == "ref") & (path.N == N)]
            g = p.groupby(["init", "iteration"]).msr_ref.mean().reset_index()
            for init, c in INIT_COLORS.items():
                x = g[g.init == init]
                ax.plot(x.iteration, 1e3 * x.msr_ref, color=c, lw=1.1)
            nm = summ[(summ.kind == "ref") & (summ.N == N) & (summ.procedure == "NPMLE")].MSR_x1e3.iloc[0]
            ax.axhline(nm, color=INK, ls="--", lw=1)
            xsym(ax, int(g.iteration.max())); ax.set_yscale("symlog", linthresh=1e-2); ax.set_ylim(bottom=0)
            ax.set_title(f"({'bc'[j]}) finite samples, N = {N:,}: MSR")
            ax.set_xlabel("EM iteration k"); ax.set_ylabel("MSR x 1e3 (P)")
    if per is not None:
        for j, N in enumerate(sorted(per.N.unique())):
            ax = fig.add_subplot(gs[1, 0 + j * 1 if j == 0 else 1:3] if False else gs[1, j if j < 2 else 2])
            x = per[(per.procedure == "EM55") & (per.N == N)]
            rng = np.random.default_rng(0)
            for i, (init, c) in enumerate(INIT_COLORS.items()):
                y = x[x.init == init].diff_x1e3.to_numpy()
                ax.scatter(i + rng.uniform(-0.25, 0.25, len(y)), y, s=8, color=c, edgecolor="white", lw=0.4)
                ax.plot([i - 0.3, i + 0.3], [np.mean(y)] * 2, color=INK, lw=1.2)
            ax.axhline(0, color=MUTED, lw=0.8)
            ax.set_xticks(range(8)); ax.set_xticklabels(list(INIT_COLORS), rotation=45, fontsize=6)
            ax.set_yscale("symlog", linthresh=0.1)
            ax.set_title(f"({'de'[j]}) per configuration, N = {N:,}:\nMSR(EM55) - MSR(NPMLE), x1e3")
        ax = fig.add_subplot(gs[1, 2])
        x = per[(per.procedure == "EM_cap")]
        for j, N in enumerate(sorted(x.N.unique())):
            for i, (init, c) in enumerate(INIT_COLORS.items()):
                y = x[(x.init == init) & (x.N == N)].diff_x1e3.mean()
                ax.bar(i + (j - 0.5) * 0.38, y, width=0.36, color=c, alpha=0.55 if j == 0 else 1.0,
                       edgecolor="white", lw=0.5)
        ax.axhline(0, color=MUTED, lw=0.8)
        ax.set_xticks(range(8)); ax.set_xticklabels(list(INIT_COLORS), rotation=45, fontsize=6)
        ax.set_title("(f) EM at cap - NPMLE, mean MSR diff x1e3\n(light: N=1e3, dark: N=1e5)")
    fig.tight_layout()
    save(fig, "fig2_initialization")


# ------------------------------------------------------------------ Fig 3
def fig3(pop_sub, finite_label):
    pdir = OUT / pop_sub
    dec = pd.read_csv(pdir / "tau2_G200_continuous_decisions.csv.gz")
    cur = pd.read_csv(pdir / "tau2_G200_continuous_curves.csv.gz")
    from experiments.em_mechanism.pairs import load_pairs
    from experiments.em_mechanism.common import gamma1_cells
    cell = next(c for c in gamma1_cells() if c.dist_name == PAIR_DIST and c.T == PAIR_T and c.beta == PAIR_BETA)
    pair = next(p for p in load_pairs(2) if p["config_id"] == cell.cid)
    W_ref = cell.population().W
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.0))
    ax = axes[0, 0]
    d = dec[(dec.distribution == PAIR_DIST) & (dec["T"] == PAIR_T) & (dec.beta == PAIR_BETA)]
    de = d[d.init == "uniform"].sort_values("iteration")
    cap = int(de.iteration.max())
    dn = d[d.init == "npmle"].iloc[0]
    for k_, (c, ls, lab) in TRUTH_STYLE.items():
        ax.plot(de.iteration, de[f"regret_{k_}"], color=c, ls=ls, label=lab)
        ax.plot([cap * 1.6], [dn[f"regret_{k_}"]], marker="D", ms=4, color=c, clip_on=False)
    xsym(ax, cap * 1.7)
    ax.set_xlabel("EM iteration k (exact probabilities, uniform start)"); ax.set_ylabel("relative regret")
    ax.set_title(f"(a) one trajectory, three compatible populations\n{FAM_LABEL[PAIR_DIST]}, T={PAIR_T}, beta={PAIR_BETA}; diamonds: NPMLE")
    ax.legend(frameon=False)
    ax = axes[0, 1]
    s = np.arange(1, PAIR_T)
    for k_, W in (("ref", W_ref), ("lower", np.asarray(pair["welfare"][0])), ("upper", np.asarray(pair["welfare"][1]))):
        c, ls, lab = TRUTH_STYLE[k_]
        ax.plot(s, W / W.max(), color=c, ls=ls, lw=1.8, label=f"true: {lab}")
    cc = cur[(cur.distribution == PAIR_DIST) & (cur["T"] == PAIR_T) & (cur.beta == PAIR_BETA)]
    for (init, k, lab, col, mk) in (("uniform", 0, "EM k=0", "#c3c2b7", "o"), ("uniform", 55, "EM k=55", INK, "s"),
                                    ("uniform", cap, f"EM k={cap:.0e}", "#4a3aa7", "^"), ("npmle", -1, "NPMLE", "#e34948", "D")):
        x = cc[(cc.init == init) & (cc.iteration == k)].sort_values("s")
        if len(x):
            ax.plot(x.s, x.W_hat / x.W_hat.max(), color=col, lw=0.9, marker=mk, ms=3, label=f"estimated: {lab}")
    ax.set_xlabel("allocation date s"); ax.set_ylabel("welfare / own maximum")
    ax.set_title("(b) true and estimated welfare curves (same pair)")
    ax.legend(frameon=False, fontsize=5.5)
    path = read(f"{finite_label}_path_by_config.csv.gz")
    risk = read(f"{finite_label}_pair_risk.csv")
    ax = axes[1, 0]
    if path is not None and (path.kind == "pair").any():
        p = path[(path.kind == "pair") & (path.init == "uniform")]
        for N, ls in zip(sorted(p.N.unique()), ("-", "--")):
            x = p[p.N == N].groupby(["iteration", "config_id"])[["mean_regret_lower", "mean_regret_upper"]].mean()
            rho = x.max(axis=1).groupby("iteration").mean()
            ax.plot(rho.index, rho.values, color=INIT_COLORS["uniform"], ls=ls, label=f"EM, N={N:,}")
            r = risk[(risk.N == N) & (risk.procedure == "NPMLE")].rho_max_of_member_means.iloc[0]
            ax.axhline(r, color=INK, ls=ls, lw=0.9, label=f"NPMLE, N={N:,}")
        xsym(ax, int(p.iteration.max()))
        ax.set_xlabel("EM iteration k"); ax.set_ylabel("pooled worst-member mean regret")
        ax.set_title("(c) 30 compatible pairs, shared pre-failure data")
        ax.legend(frameon=False, fontsize=5.5)
    ax = axes[1, 1]
    pos = read("population_identified_interval_position.csv.gz")
    if pos is not None:
        pos = pos[(pos.tau == 2) & (pos.G == 200)]
        capp = int(pos.iteration.max())
        for (init, k, lab, col, mk) in (("uniform", 55, "EM k=55", INK, "s"), ("uniform", capp, f"EM k={capp:.0e}", "#4a3aa7", "^"),
                                        ("npmle", -1, "NPMLE", "#e34948", "D")):
            x = pos[(pos.init == init) & (pos.iteration == k)]
            ax.scatter(x.position_P, x.position, s=12, marker=mk, color=col, label=lab, edgecolor="white", lw=0.4)
        lim = [-0.1, 1.1]
        ax.plot(lim, lim, color=MUTED, lw=0.7)
        ax.axhline(0, color=TRUTH_STYLE["lower"][0], ls="--", lw=0.8)
        ax.axhline(1, color=TRUTH_STYLE["upper"][0], ls=":", lw=0.8)
        ax.set_xlim(-0.05, 0.6); ax.set_ylim(-0.6, 1.6)
        ax.set_xlabel("position of P in the identified interval")
        ax.set_ylabel("position of the estimate")
        ax.set_title("(d) horizon survival E[(1-p)^(T-1)]: 0 = lower,\n1 = upper member (exact probabilities)")
        ax.legend(frameon=False, fontsize=5.5)
    fig.tight_layout()
    save(fig, "fig3_compatible_populations")


# ------------------------------------------------------------------ Fig 4
def fig4(pop_sub):
    import glob
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.4))
    rows = []
    for f in glob.glob(str(OUT / pop_sub / "*_decisions.csv.gz")):
        d = pd.read_csv(f)
        rows.append(d[(d.init.isin(["uniform", "npmle"]))])
    if not rows:
        return
    d = pd.concat(rows)
    for ax, (title, sel, key) in zip(axes, (
            ("(a) grid resolution (tau = 2)", (d.tau == 2) & (d.variant == "continuous"), "G"),
            ("(b) observation length (G = 200)", (d.G == 200) & (d.variant == "continuous"), "tau"),
            ("(c) continuous vs grid-realizable truth", (d.tau == 2) & (d.G == 200), "variant"))):
        x = d[sel]
        for i, (v, g) in enumerate(x.groupby(key)):
            col = list(INIT_COLORS.values())[i]
            e = g[g.init == "uniform"]
            rc = "regret_realizable" if (key == "variant" and v == "realizable") else "regret_ref"
            m = e.groupby("iteration")[rc].mean()
            ax.plot(m.index, m.values, color=col, label=f"{key}={v}")
            ax.axhline(g[g.init == "npmle"][rc].mean(), color=col, ls="--", lw=0.9)
        xsym(ax, int(x.iteration.max())); ax.set_yscale("symlog", linthresh=1e-4); ax.set_ylim(bottom=0)
        ax.set_title(title); ax.set_xlabel("EM iteration k")
        ax.legend(frameon=False, fontsize=5.5)
    axes[0].set_ylabel("mean regret (dashed: NPMLE)")
    fig.tight_layout()
    save(fig, "fig4_grid_and_prefix")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--finite", default="finite")
    ap.add_argument("--population", default="population")
    a = ap.parse_args(argv)
    fig1(a.population, a.finite)
    fig2(a.population, a.finite)
    fig3(a.population, a.finite)
    fig4(a.population)
    print(f"figures -> {FIG}")


if __name__ == "__main__":
    main()
