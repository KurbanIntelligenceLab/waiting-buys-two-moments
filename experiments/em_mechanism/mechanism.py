"""Mechanism follow-up (MECHANISM_PROTOCOL.md): what continued EM changes in the
estimated risk distribution, how that change moves the welfare contrast between
the competing dates, and whether constraining it removes the deterioration.

Inputs: the completed finite-sample trajectories (weight snapshots in
runs/em_mechanism/finite/*.npz, dense dates in results/em_mechanism/finite/cells/)
and the exact-probability snapshots (runs/em_mechanism/population/).

Usage:  python -m experiments.em_mechanism.mechanism [--workers 4]
Writes  results/em_mechanism/mechanism/
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linprog

from experiments.em_mechanism.common import OUT, gamma1_cells, reference_regret, write_csv_atomic

MOUT = OUT / "mechanism"
BINS = np.array([0.0, 0.02, 0.05, 0.10, 0.20, 0.40, 1.0 + 1e-12])
BIN_LABELS = ["[0,.02)", "[.02,.05)", "[.05,.1)", "[.1,.2)", "[.2,.4)", "[.4,1]"]
N_PATH = 400
CAP = 1_000_000
INIT_INDEX_UNIFORM = 0
N_INITS = 8


# ------------------------------------------------------------------ welfare and its gradient
def welfare_and_grad(w, PHI, UTIL, beta):
    """W^s(w) and its (envelope) gradient in w: sum_k a_k (UTIL_k - lambda PHI_k)."""
    m, v = PHI @ w, UTIL @ w
    rem, W = beta, 0.0
    a = np.zeros(len(m))
    lam = 0.0
    for k in range(len(m) - 1, -1, -1):
        if rem <= 0.0:
            break
        if m[k] <= 0.0:
            continue
        take = min(rem, m[k])
        a[k] = take / m[k]
        W += a[k] * v[k]
        rem -= take
        if a[k] < 1.0:
            lam = v[k] / m[k]
    g = a @ UTIL - lam * (a @ PHI)
    return W, g


def contrast(w, ev, d_a, d_b, scale):
    """C(w) = [W^{d_b}(w) - W^{d_a}(w)] / scale and its gradient."""
    Pa, Ua = ev.cache[d_a - 1]
    Pb, Ub = ev.cache[d_b - 1]
    Wa, ga = welfare_and_grad(w, Pa, Ua, ev.beta)
    Wb, gb = welfare_and_grad(w, Pb, Ub, ev.beta)
    return (Wb - Wa) / scale, (gb - ga) / scale


def path_attribution(w0, w1, ev, d_a, d_b, scale, grid):
    """Attribute C(w1) - C(w0) to grid points along the straight path (midpoint rule)."""
    delta = w1 - w0
    gbar = np.zeros_like(w0)
    for j in range(N_PATH):
        t = (j + 0.5) / N_PATH
        gbar += contrast(w0 + t * delta, ev, d_a, d_b, scale)[1]
    gbar /= N_PATH
    contrib = delta * gbar
    exact = contrast(w1, ev, d_a, d_b, scale)[0] - contrast(w0, ev, d_a, d_b, scale)[0]
    by_bin = np.array([contrib[(grid >= BINS[b]) & (grid < BINS[b + 1])].sum() for b in range(len(BIN_LABELS))])
    return exact, by_bin, float(exact - contrib.sum())


# ------------------------------------------------------------------ distribution features
def features(w, grid, Psi, q_emp, q_true, prefix):
    ent = -np.sum(w[w > 0] * np.log(w[w > 0]))
    srt = np.sort(w)[::-1]
    qh = Psi @ w
    f = {f"{prefix}_eff_support": float(np.exp(ent)),
         f"{prefix}_n99": int(np.searchsorted(np.cumsum(srt), 0.99) + 1),
         f"{prefix}_kl_emp": (float(np.sum(np.where(q_emp > 0, q_emp * np.log(np.where(q_emp > 0, q_emp, 1) / qh), 0.0)))
                              if q_emp is not None else np.nan),
         f"{prefix}_kl_true": float(np.sum(q_true * np.log(q_true / qh)))}
    for j in range(1, 11):
        f[f"{prefix}_m{j}"] = float(w @ grid ** j)
    for b, lab in enumerate(BIN_LABELS):
        f[f"{prefix}_mass{b}"] = float(w[(grid >= BINS[b]) & (grid < BINS[b + 1])].sum())
    return f


# ------------------------------------------------------------------ constrained alternatives (LP)
def l1_project(target, A_eq, b_eq):
    """min ||w - target||_1  s.t.  A_eq w = b_eq, w >= 0.  Returns (w, max residual) or (None, msg)."""
    G = len(target)
    I = np.eye(G)
    res = linprog(np.concatenate([np.zeros(G), np.ones(G)]),
                  A_ub=np.block([[I, -I], [-I, -I]]), b_ub=np.concatenate([target, -target]),
                  A_eq=np.hstack([A_eq, np.zeros((A_eq.shape[0], G))]), b_eq=b_eq,
                  bounds=[(0, None)] * (2 * G), method="highs",
                  options={"primal_feasibility_tolerance": 1e-10, "dual_feasibility_tolerance": 1e-10})
    if not res.success:
        return None, res.message
    w = np.maximum(res.x[:G], 0.0)
    return w, float(np.abs(A_eq @ w - b_eq).max())


def moment_rows(grid, js, w_ref):
    rows = np.vstack([grid ** j for j in js])
    b = rows @ w_ref
    s = np.where(np.abs(b) > 0, np.abs(b), 1.0)
    return rows / s[:, None], b / s


def bin_rows(grid, bins_idx, w_ref):
    rows = np.vstack([((grid >= BINS[b]) & (grid < BINS[b + 1])).astype(float) for b in bins_idx])
    return rows, rows @ w_ref


# ------------------------------------------------------------------ local switch search
def find_switches(counts, w55, d55, brackets, Psi, ev):
    """Re-run EM from w55 (iteration 55) with fit_npmle's update and the 1e-300 floor,
    checking the date every iteration inside each row's bracket [lo, hi].
    Returns per row (k_star, w_{k*-1}, w_{k*}, date_after, date_at_lo)."""
    n = np.asarray(counts, dtype=float)
    N = n.sum(axis=1, keepdims=True)
    W = np.array(w55, dtype=float)
    PsiT = np.ascontiguousarray(Psi.T)
    B = W.shape[0]
    lo = np.array([b[0] for b in brackets]); hi = np.array([b[1] for b in brackets])
    out = [None] * B
    date_lo = np.full(B, -1)
    prev = W.copy()
    k = 55
    kmax = int(hi.max())
    while True:
        act = np.flatnonzero((lo <= k) & (k <= hi) & np.array([o is None for o in out]))
        if len(act):
            C = ev.curves(W[act])
            d = ev.decide(W[act], C)
            for r, dd in zip(act, d):
                if k == lo[r]:
                    date_lo[r] = dd
                if dd != d55[r] and k > lo[r]:
                    out[r] = (k, prev[r].copy(), W[r].copy(), int(dd))
        if k >= kmax or all(o is not None for o in out):
            break
        prev[:] = W
        mix = np.maximum(W @ PsiT, 1e-300)
        W *= (n / mix) @ Psi
        W /= N
        W /= W.sum(axis=1, keepdims=True)
        np.maximum(W, 1e-300, out=W)
        k += 1
    return out, date_lo


# ------------------------------------------------------------------ per cell
def run_cell(args):
    key, events = args
    from experiments.em_mechanism.core import Welfare, exact_cell_probs, grid_and_psi
    from experiments.em_mechanism.pairs import load_pairs
    from experiments.em_mechanism.run_finite import simulate_counts
    meta = json.loads((OUT / "finite" / "cells" / f"{key}.done.json").read_text())["settings"]
    cell = next(c for c in gamma1_cells() if c.cid == meta["cid"])
    pair = next(p for p in load_pairs(2) if p["config_id"] == cell.cid)
    truths = {"ref": reference_regret(cell), "lower": np.asarray(pair["regret"][0]),
              "upper": np.asarray(pair["regret"][1])}
    Wstar = cell.population().W_star
    grid, Psi = grid_and_psi(200, 2)
    ev = Welfare(grid, cell.T, cell.beta)
    q_true = exact_cell_probs(cell.dist, 2)
    seeds = meta["seeds"]
    counts, _ = simulate_counts("ref", cell, pair, meta["N"], seeds)
    from experiments.em_mechanism.snapshots import finite_cell
    snap = finite_cell("finite", key)
    dz = np.load(OUT / "finite" / "cells" / f"{key}_dense.npz")
    iters = dz["iters"]
    pos = {s: i for i, s in enumerate(seeds)}
    rows = []
    sel = [pos[e["seed"]] for e in events]
    w55 = snap.at(55)[sel]
    wcap = snap.at(CAP)[sel]
    dense_col = {s: j for j, s in enumerate(dz["seeds"]) if dz["inits"][j] == "uniform"}
    d55 = np.array([e["d55"] for e in events])
    # brackets for the first post-55 switch
    brackets, need = [], []
    for e in events:
        col = dz["dates"][:, dense_col[e["seed"]]]
        r = int(np.searchsorted(iters, 55))
        later = np.flatnonzero((iters > 55) & (col != e["d55"]))
        if len(later):
            j = later[0]
            brackets.append((int(iters[j - 1]), int(iters[j]))); need.append(True)
        else:
            brackets.append((55, 55)); need.append(False)
    idx_need = [i for i, f in enumerate(need) if f]
    switches = [None] * len(events)
    date_lo = np.full(len(events), -1)
    if idx_need:
        sw, dl = find_switches(counts[[sel[i] for i in idx_need]], w55[idx_need], d55[idx_need],
                               [brackets[i] for i in idx_need], Psi, ev)
        for a, i in enumerate(idx_need):
            switches[i] = sw[a]; date_lo[i] = dl[a]

    for i, e in enumerate(events):
        s_i = sel[i]
        q_emp = counts[s_i] / counts[s_i].sum()
        a, b = w55[i], wcap[i]
        dA, dB = int(e["d55"]), int(e["dcap"])
        row = {"key": key, "config_id": cell.cid, "distribution": cell.dist_name, "T": cell.T,
               "beta": cell.beta, "N": meta["N"], "seed_index": e["seed"], "cls": e["cls"],
               "d55": dA, "dcap": dB, "r55": e["r55"], "rcap": e["rcap"]}
        row.update(features(a, grid, Psi, q_emp, q_true, "w55"))
        row.update(features(b, grid, Psi, q_emp, q_true, "wcap"))
        row["max_abs_dq"] = float(np.abs(Psi @ (b - a)).max())
        if dA != dB:
            C0 = contrast(a, ev, dA, dB, Wstar)[0]
            C1 = contrast(b, ev, dA, dB, Wstar)[0]
            exact, by_bin, resid = path_attribution(a, b, ev, dA, dB, Wstar, grid)
            row.update({"C55": C0, "Ccap": C1, "dC": exact, "attr_residual": resid,
                        **{f"attr{bb}": by_bin[bb] for bb in range(len(BIN_LABELS))}})
            wp, res = l1_project(a, Psi, Psi @ b)
            if wp is not None:
                row.update({"C_obs_part": contrast(wp, ev, dA, dB, Wstar)[0] - C0,
                            "C_fiber_part": C1 - contrast(wp, ev, dA, dB, Wstar)[0],
                            "proj_resid": res, "proj_l1_from_w55": float(np.abs(wp - a).sum()),
                            "cap_l1_from_w55": float(np.abs(b - a).sum())})
        # local switch
        sw = switches[i]
        if sw is not None:
            k_star, wpre, wpost, d_after = sw
            row.update({"k_star": k_star, "bracket_lo": brackets[i][0], "bracket_hi": brackets[i][1],
                        "date_at_lo_check": int(date_lo[i]), "d_after_switch": d_after})
            ex, bb_, rs = path_attribution(wpre, wpost, ev, dA, d_after, Wstar, grid)
            row.update({"local_dC": ex, "local_attr_residual": rs,
                        **{f"local_attr{k}": bb_[k] for k in range(len(BIN_LABELS))},
                        **{f"local_dmass{k}": float(((grid >= BINS[k]) & (grid < BINS[k + 1])) @ (wpost - wpre))
                           for k in range(len(BIN_LABELS))},
                        **{f"local_dm{j}": float((wpost - wpre) @ grid ** j) for j in range(1, 11)},
                        "local_max_abs_dq": float(np.abs(Psi @ (wpost - wpre)).max())})
        rows.append((row, a, b, s_i))
    return key, rows


# ------------------------------------------------------------------ constraint tests (after bin selection)
def constraint_tests(args):
    key, items, top_bins, low_bins = args
    from experiments.em_mechanism.core import Welfare, grid_and_psi
    from experiments.em_mechanism.pairs import load_pairs
    meta = json.loads((OUT / "finite" / "cells" / f"{key}.done.json").read_text())["settings"]
    cell = next(c for c in gamma1_cells() if c.cid == meta["cid"])
    pair = next(p for p in load_pairs(2) if p["config_id"] == cell.cid)
    truths = {"ref": reference_regret(cell), "lower": np.asarray(pair["regret"][0]),
              "upper": np.asarray(pair["regret"][1])}
    grid, Psi = grid_and_psi(200, 2)
    ev = Welfare(grid, cell.T, cell.beta)
    out = []
    for row, a, b in items:
        if row["d55"] == row["dcap"]:
            continue
        qb, qa = Psi @ b, Psi @ a
        tests = {}
        tests["T1_minimal_change"] = l1_project(a, Psi, qb)
        for J in (6, 8):
            R, r = moment_rows(grid, range(4, J + 1), a)
            tests[f"T2_freeze_m4_m{J}"] = l1_project(b, np.vstack([Psi, R]), np.concatenate([qb, r]))
        R, r = bin_rows(grid, top_bins, a)
        tests["T3_freeze_decision_bins"] = l1_project(b, np.vstack([Psi, R]), np.concatenate([qb, r]))
        R, r = bin_rows(grid, low_bins, a)
        tests["T4_freeze_irrelevant_bins"] = l1_project(b, np.vstack([Psi, R]), np.concatenate([qb, r]))
        R, r = bin_rows(grid, top_bins, b)
        tests["T5_import_decision_bins"] = l1_project(a, np.vstack([Psi, R]), np.concatenate([qa, r]))
        o = {k: row[k] for k in ("key", "config_id", "distribution", "N", "seed_index", "cls", "d55", "dcap",
                                 "r55", "rcap")}
        for name, (w, res) in tests.items():
            if w is None:
                o[f"{name}_feasible"] = False
                continue
            d = int(ev.decide(w[None], ev.curves(w[None]))[0])
            o.update({f"{name}_feasible": True, f"{name}_resid": res, f"{name}_date": d,
                      **{f"{name}_regret_{t}": float(R_[d - 1]) for t, R_ in truths.items()}})
        o.update({f"r55_{t}": float(R_[row['d55'] - 1]) for t, R_ in truths.items()})
        o.update({f"rcap_{t}": float(R_[row['dcap'] - 1]) for t, R_ in truths.items()})
        out.append(o)
    return out


def population_control():
    """Exact probabilities, tau = 2, G = 200: steps 1-2 for 55 -> cap wherever the date changes."""
    from experiments.em_mechanism.core import Welfare, exact_cell_probs, grid_and_psi
    from experiments.em_mechanism.snapshots import population_task
    task = population_task("population", "tau2_G200_continuous")
    cap = task.cap
    grid, Psi = grid_and_psi(200, 2)
    meta = task.rows
    rows = []
    for cell in gamma1_cells():
        ev = Welfare(grid, cell.T, cell.beta)
        R = reference_regret(cell)
        q_true = exact_cell_probs(cell.dist, 2)
        Wstar = cell.population().W_star
        for i, (dn, init) in enumerate(meta):
            if dn != cell.dist_name:
                continue
            a, b = task.snaps[55][i], task.snaps[cap][i]
            dA = int(ev.decide(a[None], ev.curves(a[None]))[0])
            dB = int(ev.decide(b[None], ev.curves(b[None]))[0])
            if dA == dB:
                continue
            exact, by_bin, resid = path_attribution(a, b, ev, dA, dB, Wstar, grid)
            r = {"distribution": dn, "T": cell.T, "beta": cell.beta, "init": init, "d55": dA, "dcap": dB,
                 "r55": R[dA - 1], "rcap": R[dB - 1], "dC": exact, "attr_residual": resid,
                 "cls": "worse" if R[dB - 1] > R[dA - 1] + 1e-12 else ("improve" if R[dB - 1] < R[dA - 1] - 1e-12 else "same"),
                 **{f"attr{bb}": by_bin[bb] for bb in range(len(BIN_LABELS))}}
            r.update(features(a, grid, Psi, None, q_true, "w55"))
            r.update(features(b, grid, Psi, None, q_true, "wcap"))
            rows.append(r)
    return pd.DataFrame(rows)


def load_events():
    import glob
    t = pd.concat([pd.read_csv(f) for f in glob.glob(str(OUT / "finite" / "cells" / "ref_*_table.csv.gz"))])
    e = t[(t.method == "em") & (t.init == "uniform") & t.iteration.isin([55, CAP])]
    w = e.pivot_table(index=["config_id", "N", "seed_index"], columns="iteration",
                      values=["date", "regret_ref"]).reset_index()
    w.columns = ["config_id", "N", "seed", "d55", "dcap", "r55", "rcap"]
    d = w.rcap - w.r55
    w["cls"] = np.where(d > 1e-12, "worse", np.where(d < -1e-12, "improve", "same"))
    return w


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out-dir", default=None,
                    help="write outputs here instead of results/em_mechanism/mechanism (reproduction checks)")
    a = ap.parse_args(argv)
    global MOUT
    if a.out_dir:
        MOUT = Path(a.out_dir).resolve()
    t0 = time.perf_counter()
    MOUT.mkdir(parents=True, exist_ok=True)
    ev_df = load_events()
    ev_df = ev_df[ev_df.cls.isin(["worse", "improve"])]
    tasks = []
    for (cid, N), g in ev_df.groupby(["config_id", "N"]):
        key = f"ref_{cid}_N{N}_G200"
        tasks.append((key, g.to_dict(orient="records")))
    print(f"{len(ev_df)} events in {len(tasks)} cells", flush=True)
    results = {}
    with mp.Pool(a.workers) as pool:
        for i, (key, rows) in enumerate(pool.imap_unordered(run_cell, tasks), 1):
            results[key] = rows
            if i % 10 == 0:
                print(f"[{i}/{len(tasks)}] {(time.perf_counter() - t0) / 60:.1f} min", flush=True)
    events = pd.DataFrame([r for rows in results.values() for r, *_ in rows])
    write_csv_atomic(events, MOUT / "events.csv.gz")

    # bin selection for T3/T4 from estimated-welfare attribution only (worse events, per N)
    sel = {}
    for N, g in events[(events.cls == "worse") & events.dC.notna()].groupby("N"):
        mag = np.array([g[f"attr{b}"].abs().mean() for b in range(len(BIN_LABELS))])
        order = np.argsort(-mag)
        sel[int(N)] = {"top": sorted(order[:2].tolist()), "low": sorted(order[-2:].tolist()),
                       "mean_abs_attr": mag.tolist()}
    (MOUT / "bin_selection.json").write_text(json.dumps(
        {str(k): {**v, "labels_top": [BIN_LABELS[b] for b in v["top"]],
                  "labels_low": [BIN_LABELS[b] for b in v["low"]]} for k, v in sel.items()}, indent=1) + "\n")
    ctasks = []
    for key, rows in results.items():
        N = rows[0][0]["N"]
        ctasks.append((key, [(r, wa, wb) for r, wa, wb, _ in rows], sel[N]["top"], sel[N]["low"]))
    with mp.Pool(a.workers) as pool:
        tests = [o for out in pool.imap_unordered(constraint_tests, ctasks) for o in out]
    write_csv_atomic(pd.DataFrame(tests), MOUT / "constraint_tests.csv.gz")
    pc = population_control()
    write_csv_atomic(pc, MOUT / "population_control.csv.gz")
    print(f"done in {(time.perf_counter() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
