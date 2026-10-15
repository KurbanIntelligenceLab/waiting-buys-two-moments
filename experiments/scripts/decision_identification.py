"""Decision-identification experiment: 30 gamma=1 configurations x tau = 2, 3, 4.

For each configuration-prefix pair, find the exact argmax set D*(P), then for
every d in D*(P) and every competitor r, either certify V(d, r) < 0 on all of
M_tau or exhibit an exact witness with V(d, r) > 0 (REDUCTION.md, Sections
6-7).  Every result is written to its own file, so a run resumes where it
stopped.

    python experiments/scripts/decision_identification.py pilot
    python experiments/scripts/decision_identification.py full [--workers 4]

Outputs (all under results/decision_identification/):
    raw/<run>/<cid>_tau<k>.json.gz   one job: summary, per-pair results,
                                     certificates and witnesses (exact)
    <run>_pairs.csv, <run>_jobs.csv  flat tables built from raw/
"""

from __future__ import annotations

import argparse
import gzip
import json
import multiprocessing as mp
import sys
import time
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from wtm.config import config_id, population_cells  # noqa: E402
from wtm.decision_id import DecisionProblem, bound_pair  # noqa: E402

OUT = ROOT / "results" / "decision_identification"

# The configurations that neither the reachable-date search nor the
# extreme-pair LP had shown to be non-unique (Table 5's "upper bound" row):
# results/unique_date_reconciliation_report.md.  Recorded for comparison only;
# the experiment runs every configuration-prefix pair regardless.
PRIOR_OPEN = {
    2: {"9fe557d42d19"},
    3: {"08d96ab0defb", "c8697d1800b5", "9fe557d42d19", "82d22f34a4dc", "48992d9e3b43",
        "9d58c5dce3c9", "fae85d196c89", "51c48458632c", "fdeab5019bb9", "eeb7694df15d",
        "1bb4f8059848", "76324efe4052", "d3ad67d001b3", "6a778be2d26d", "f3e14f3cb88f",
        "a22714b128d1", "0f904f38c8bf", "06380b49a6f3"},
    4: {"08d96ab0defb", "c8697d1800b5", "9fe557d42d19", "82d22f34a4dc", "82fa2d570256",
        "fad987cc30ad", "645ba2bddb66", "48992d9e3b43", "9d58c5dce3c9", "fae85d196c89",
        "51c48458632c", "fdeab5019bb9", "6ee1a95c2835", "d157b5b6b9b0", "1bb4f8059848",
        "89a9f7e78122", "9cc5ff7a0659", "76324efe4052", "d3ad67d001b3", "6eb1e1952502",
        "6a778be2d26d", "f3e14f3cb88f", "7a44ab9f6ee6", "a22714b128d1", "0f904f38c8bf",
        "06380b49a6f3", "25798ac797cd"},
}

MAX_BOXES = 3000
TIME_LIMIT = 900.0


def fr(x: Fraction) -> str:
    return f"{x.numerator}/{x.denominator}"


def fl(x: Fraction) -> str:
    """Exact hex form of a float-derived (dyadic) rational."""
    return float(x).hex() if Fraction(float(x)) == x else fr(x)


def cells_gamma1():
    cells = [c for c in population_cells() if c.gamma == 1]
    assert len(cells) == 30
    return {config_id(dist=c.dist_name, gamma=1, T=c.T, beta=c.beta): c for c in cells}


def pair_record(res, W, d, prob):
    rec = {"d": d, "r": res.r, "status": res.status, "n_boxes": res.n_boxes,
           "n_leaves": res.n_leaves, "seconds": round(res.seconds, 3), "note": res.note,
           "truth_gap_exact": fr(W[res.r] - W[d]),
           "truth_gap_rel": float((W[res.r] - W[d]) / W[d]),
           "worst_certified_U_over_beta": res.worst_certified_U,
           "worst_certified_U_rel": (res.worst_certified_U * prob.beta / float(W[d])
                                     if res.n_leaves else None),
           "open_float_upper_over_beta": res.max_float_upper}
    if res.status == "certified":
        rec["certificate"] = [
            {"box": [fr(b) for b in box], "U": fr(U),
             "y": [fl(v) for v in parts["y"]], "eta": fl(parts["eta"]),
             "omega": [fl(v) for v in parts["omega"]]}
            for box, U, parts in res.leaves]
    if res.witness is not None:
        wit = res.witness
        rec["witness"] = {"atoms": [fl(a) for a in wit["atoms"]],
                          "weights": [fl(w) for w in wit["weights"]],
                          "eps": fl(wit["eps"]), "g": [fr(c) for c in wit["g"]],
                          "W_r": fr(wit["W_r"]), "W_d": fr(wit["W_d"])}
        rec["witness_advantage_rel"] = float(wit["advantage"] / W[d])
    return rec


def classify(D, pairs):
    """REDUCTION.md, Section 7."""
    by_d = {d: [p for p in pairs if p["d"] == d] for d in D}
    for d in D:
        if by_d[d] and all(p["status"] == "certified" for p in by_d[d]):
            return "CERTIFIED_UNIQUE", d
    if len(D) >= 2:
        return "TIE_AT_TRUTH", None
    if all(any(p["status"] == "witness" for p in by_d[d]) for d in D):
        return "NO_GUARANTEED_DATE", None
    return "UNDECIDED", None


def run_job(args):
    cid, tau, run, opts = args
    path = OUT / "raw" / run / f"{cid}_tau{tau}.json.gz"
    if path.exists():
        return cid, tau, "skipped"
    cell = cells_gamma1()[cid]
    t0 = time.time()
    first_date = tau if opts.get("actionable") else 1
    prob = DecisionProblem(cell.dist, cell.T, cell.beta, tau, first_date=first_date)
    D, W = prob.truth_argmax()
    pairs = []
    for d in D:
        # closest competitors first: they are the likeliest to yield a witness
        comps = sorted((r for r in prob.dates if r != d), key=lambda r: W[d] - W[r])
        for r in comps:
            if r in D:     # an exact tie at the truth: P itself is the tie witness
                pairs.append({"d": d, "r": r, "status": "tie_at_truth",
                              "truth_gap_exact": "0/1"})
                continue
            res = bound_pair(prob, d, r, max_boxes=opts.get("max_boxes", MAX_BOXES),
                             time_limit=opts.get("time_limit", TIME_LIMIT),
                             search_witness=opts.get("search_witness", True))
            pairs.append(pair_record(res, W, d, prob))
            if res.status == "witness":
                break      # d is refuted; remaining competitors are not needed
    outcome, date = classify(D, pairs)
    job = {"config_id": cid, "distribution": cell.dist_name, "T": cell.T,
           "beta": cell.beta, "tau": tau, "n_moments": 2 * tau - 1,
           "first_date": first_date, "candidate_dates": prob.dates,
           "truth_argmax": D, "truth_welfare": {str(s): fr(W[s]) for s in prob.dates},
           "prior_status": "open" if cid in PRIOR_OPEN[tau] else "non_unique_witnessed",
           "outcome": outcome, "identified_date": date,
           "n_pairs": len(pairs), "seconds": round(time.time() - t0, 2),
           "options": opts, "pairs": pairs}
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        json.dump(job, fh)
    return cid, tau, outcome


def run(jobs, run_name, workers, opts):
    work = [(cid, tau, run_name, opts) for cid, tau in jobs]
    t0 = time.time()
    if workers <= 1:
        for w in work:
            print(*run_job(w), f"{time.time() - t0:.0f}s", flush=True)
    else:
        with mp.Pool(workers) as pool:
            for out in pool.imap_unordered(run_job, work):
                print(*out, f"{time.time() - t0:.0f}s", flush=True)


PILOT = [
    # Theorem 1's reported case (Beta(1,32), T=20, beta=0.01): must be refuted
    ("fad987cc30ad", 2),
    # search false-positives that the extreme-pair LP refuted: must be refuted
    ("08d96ab0defb", 2), ("645ba2bddb66", 3), ("eeb7694df15d", 4),
    # the only configuration still open at tau = 2
    ("9fe557d42d19", 2),
    # open at tau = 4, one per distribution family, for timing
    ("06380b49a6f3", 4), ("89a9f7e78122", 4), ("d3ad67d001b3", 4), ("9fe557d42d19", 4),
]


def extend_job(args):
    """Re-run only the undecided pairs of an undecided job with a larger budget.

    The main-run file is left untouched; the combined result is written to
    raw/<run>/ and reported separately from the main run.
    """
    src, run, opts = args
    with gzip.open(src, "rt", encoding="utf-8") as fh:
        job = json.load(fh)
    path = OUT / "raw" / run / src.name
    if path.exists():
        return job["config_id"], job["tau"], "skipped"
    cell = cells_gamma1()[job["config_id"]]
    prob = DecisionProblem(cell.dist, cell.T, cell.beta, job["tau"],
                           first_date=job.get("first_date", 1))
    D, W = prob.truth_argmax()
    t0 = time.time()
    pairs = []
    for p in job["pairs"]:
        if p["status"] != "undecided":
            pairs.append(p)
            continue
        res = bound_pair(prob, p["d"], p["r"], max_boxes=opts["max_boxes"],
                         time_limit=opts["time_limit"])
        rec = pair_record(res, W, p["d"], prob)
        rec["extended_from_main_run"] = True
        pairs.append(rec)
    outcome, date = classify(D, pairs)
    job.update({"pairs": pairs, "outcome": outcome, "identified_date": date,
                "main_run_outcome": job["outcome"], "extended_options": opts,
                "extended_seconds": round(time.time() - t0, 2)})
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        json.dump(job, fh)
    return job["config_id"], job["tau"], outcome


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["pilot", "soundness", "full", "extend", "actionable"])
    ap.add_argument("--source", default="full", help="extend: run whose undecided jobs to re-run")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--run", default=None)
    a = ap.parse_args()
    cells = cells_gamma1()
    if a.mode == "pilot":
        run(PILOT, a.run or "pilot", a.workers, {})
    elif a.mode == "soundness":
        # witness search disabled on configurations known to be ambiguous: the
        # certifier must never report "certified" for a refuted date
        run([("fad987cc30ad", 2), ("08d96ab0defb", 2), ("645ba2bddb66", 3),
             ("eeb7694df15d", 4)], a.run or "soundness", a.workers,
            {"search_witness": False, "max_boxes": 300, "time_limit": 300})
    elif a.mode == "extend":
        opts = {"max_boxes": 30000, "time_limit": 7200.0}
        src = []
        for f in sorted((OUT / "raw" / a.source).glob("*.json.gz")):
            with gzip.open(f, "rt", encoding="utf-8") as fh:
                if json.load(fh)["outcome"] == "UNDECIDED":
                    src.append(f)
        work = [(f, a.run or "extended", opts) for f in src]
        with mp.Pool(max(1, min(a.workers, len(work)))) as pool:
            for out in pool.imap_unordered(extend_job, work):
                print(*out, flush=True)
    elif a.mode == "actionable":
        # candidate dates restricted to s >= tau: is the best date that is
        # still available determined by the observations?
        jobs = [(cid, tau) for tau in (2, 3, 4) for cid in sorted(cells)]
        run(jobs, a.run or "actionable", a.workers, {"actionable": True})
    else:
        jobs = [(cid, tau) for tau in (2, 3, 4) for cid in sorted(cells)]
        run(jobs, a.run or "full", a.workers, {})


if __name__ == "__main__":
    main()
