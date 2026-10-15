"""Exact re-verification of the Table 2 certification archives.

Nothing here comes from the floating-point search, which only proposes
certificates and witnesses. This script rebuilds each problem from the frozen
configuration and checks the stored rationals exactly. Then, separately, it
decides whether each job's claimed outcome follows from the comparisons that
verified.

Per run (full, extended, actionable, actionable_extended):

  * Coverage. The archive must contain exactly the expected job keys: every
    (configuration, tau) for full and actionable; for an extension, exactly
    the jobs its source run left UNDECIDED. Missing, duplicate, unknown,
    misnamed or malformed job files are errors, so an absent archive fails
    instead of verifying zero jobs.
  * Domain. The candidate dates are {1, ..., T-1} for full and extended and
    {tau, ..., T-1} for the actionable runs. A job's first_date,
    candidate_dates, truth argmax and exact truth welfare must match the
    domain recomputed here.
  * Comparisons. Each (d, r) comparison has a known status, a reference date
    d in the exact truth argmax, a competitor r in the domain with r != d, and
    no duplicates. A certified comparison's leaves must each give U_cert < 0
    and must tile [0, r+1] x [0, 1] exactly. A witness must be an exact member
    of M_tau with W^r(Q) > W^d(Q) exactly.
  * Outcome. The outcome is re-derived from the verified comparisons alone,
    and must match the stored one:
      CERTIFIED_UNIQUE(d)   the exact truth argmax is {d}, and verified
                            certificates cover EVERY competitor in the domain;
      NO_GUARANTEED_DATE    the truth argmax is a single date d, and some
                            verified witness refutes d;
      TIE_AT_TRUTH          the truth argmax has two or more dates (exact);
      UNDECIDED             otherwise.
    A valid stored comparison is not the same as a justified outcome; both
    are reported.
  * Extensions. A job in an extension run must keep its source job's resolved
    comparisons unchanged and may only replace undecided ones.

Finally, Table 2 is rebuilt from the verified outcomes. All dates use full plus
extended; available dates use actionable plus actionable_extended. Every
(configuration, tau) key must occur exactly once, and no UNDECIDED or
TIE_AT_TRUTH outcome may remain.

    python experiments/scripts/verify_decision_certificates.py            # all four runs + Table 2
    python experiments/scripts/verify_decision_certificates.py full       # one run
    python experiments/scripts/verify_decision_certificates.py --archive DIR   # another archive root
"""
from __future__ import annotations

import argparse
import gzip
import json
import re
import sys
import time
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from wtm.config import config_id, load_frozen, population_cells  # noqa: E402
from wtm.decision_id import (DecisionProblem, bernstein_lower_bound,  # noqa: E402
                             mono_to_bernstein, verify_box_certificate,
                             welfare_exact_mixture)

DEFAULT_ARCHIVE = ROOT / "results" / "decision_identification"
TAUS = (2, 3, 4)
RUNS = {"full": None, "extended": "full",
        "actionable": None, "actionable_extended": "actionable"}
ACTIONABLE = {"actionable", "actionable_extended"}
OUTCOMES = {"CERTIFIED_UNIQUE", "NO_GUARANTEED_DATE", "TIE_AT_TRUTH", "UNDECIDED"}
STATUSES = {"certified", "witness", "tie_at_truth", "undecided"}
REQUIRED = {"config_id": str, "tau": int, "T": int, "truth_argmax": list,
            "truth_welfare": dict, "outcome": str, "pairs": list}
WITNESS_NOTES: list = []   # witnesses whose atomic part A has mass != 1
NAME = re.compile(r"^([0-9a-f]{12})_tau([0-9]+)\.json\.gz$")


class Malformed(Exception):
    pass


def cells_gamma1():
    out = {}
    for c in population_cells(load_frozen()):
        if c.gamma == 1:
            out[config_id(dist=c.dist_name, gamma=1, T=c.T, beta=c.beta)] = c
    if len(out) != 30:
        raise RuntimeError(f"expected 30 gamma=1 configurations, found {len(out)}")
    return out


def parse(s: str) -> Fraction:
    if not isinstance(s, str):
        raise Malformed(f"expected a rational string, got {type(s).__name__}")
    if "/" in s:
        a, b = s.split("/")
        return Fraction(int(a), int(b))
    return Fraction(float.fromhex(s))


def disjoint_tiling(boxes, r) -> bool:
    """Exact check that the boxes tile [0, r+1] x [0, 1]."""
    for tl, tu, ll, lu in boxes:
        if not (0 <= tl < tu <= r + 1 and 0 <= ll < lu <= 1):
            return False
    area = sum(((tu - tl) * (lu - ll) for tl, tu, ll, lu in boxes), Fraction(0))
    if area != r + 1:
        return False
    order = sorted(boxes)
    active = []
    for b in order:
        active = [a for a in active if a[1] > b[0]]
        for a in active:
            if a[2] < b[3] and b[2] < a[3]:
                return False
        active.append(b)
    return True


def pair_structure(prob, D, p) -> list:
    """Status, date-domain and reference checks for one stored comparison."""
    d, r, st = p.get("d"), p.get("r"), p.get("status")
    if st not in STATUSES:
        return [f"unknown comparison status {st!r}"]
    if not isinstance(d, int) or not isinstance(r, int):
        return ["comparison dates are not integers"]
    errs = []
    if d not in D:
        errs.append(f"(d={d}, r={r}): reference date not in the exact truth argmax {D}")
    if r not in prob.dates or r == d:
        errs.append(f"(d={d}, r={r}): competitor outside the domain {prob.dates[0]}..{prob.dates[-1]}")
    return errs


def check_pair(prob, cell, D, W, p) -> list:
    """Exact re-check of one structurally valid comparison (empty: valid)."""
    d, r, st = p["d"], p["r"], p["status"]
    errs = []
    if st == "certified":
        leaves = p.get("certificate")
        if not leaves:
            return [f"(d={d}, r={r}): certified without certificate leaves"]
        boxes = []
        for leaf in leaves:
            box = tuple(parse(v) for v in leaf["box"])
            boxes.append(box)
            U, _, _ = verify_box_certificate(
                prob, d, r, box, [float.fromhex(v) for v in leaf["y"]],
                float.fromhex(leaf["eta"]), [float.fromhex(v) for v in leaf["omega"]])
            if not U < 0:
                errs.append(f"(d={d}, r={r}) leaf {leaf['box']}: U_cert={float(U)} >= 0")
        if not disjoint_tiling(boxes, r):
            errs.append(f"(d={d}, r={r}): leaves do not tile [0, r+1] x [0, 1]")
    elif st == "witness":
        w = p.get("witness")
        if not w:
            return [f"(d={d}, r={r}): witness status without a witness"]
        atoms = [parse(v) for v in w["atoms"]]
        wts = [parse(v) for v in w["weights"]]
        eps = parse(w["eps"])
        g = [parse(v) for v in w["g"]]
        # Q's unit mass is checked exactly as moment j = 0 below. The atomic part
        # A itself need not have mass exactly 1: its weights are float-derived
        # rationals, and the correction g absorbs the residual.
        if not (0 < eps < 1 and all(0 <= a <= 1 for a in atoms) and all(x >= 0 for x in wts)):
            errs.append(f"(d={d}, r={r}) witness: invalid atoms, weights or eps")
        if sum(wts, Fraction(0)) != 1:
            WITNESS_NOTES.append((d, r, float(sum(wts, Fraction(0)) - 1)))
        if bernstein_lower_bound(mono_to_bernstein([1 + g[0]] + g[1:])) < 0:
            errs.append(f"(d={d}, r={r}) witness: 1+g negative on [0,1]")
        for j in range(prob.n + 1):
            mA = sum((x * a ** j for a, x in zip(atoms, wts)), Fraction(0))
            mB = cell.dist.mixed_exact(j, 0) + sum(
                (c * cell.dist.mixed_exact(j + i, 0) for i, c in enumerate(g)), Fraction(0))
            if (1 - eps) * mA + eps * mB != cell.dist.mixed_exact(j, 0):
                errs.append(f"(d={d}, r={r}) witness: moment {j} not matched exactly")
        Wr = welfare_exact_mixture(prob, r, atoms, wts, eps, g)
        Wd = welfare_exact_mixture(prob, d, atoms, wts, eps, g)
        if not Wr > Wd:
            errs.append(f"(d={d}, r={r}) witness: W^r <= W^d")
        if Wr != parse(w["W_r"]) or Wd != parse(w["W_d"]):
            errs.append(f"(d={d}, r={r}) witness: stored welfare differs from exact")
    elif st == "tie_at_truth":
        if W[r] != W[d]:
            errs.append(f"(d={d}, r={r}) claimed tie at truth is not exact")
    return errs


def derive_outcome(prob, D, valid):
    """Outcome justified by verified comparisons alone (REDUCTION.md, Section 7)."""
    if len(D) >= 2:
        return "TIE_AT_TRUTH", None
    d = D[0]
    competitors = set(prob.dates) - {d}
    certified = {p["r"] for p in valid if p["d"] == d and p["status"] == "certified"}
    if competitors and competitors <= certified:
        return "CERTIFIED_UNIQUE", d
    if any(p["d"] == d and p["status"] == "witness" for p in valid):
        return "NO_GUARANTEED_DATE", None
    return "UNDECIDED", None


def verify_job(job, run, cells) -> dict:
    """Errors split into stored-comparison errors and outcome-justification errors."""
    for k, typ in REQUIRED.items():
        if k not in job or not isinstance(job[k], typ):
            raise Malformed(f"field {k!r} missing or not {typ.__name__}")
    if job["outcome"] not in OUTCOMES:
        raise Malformed(f"unknown outcome {job['outcome']!r}")
    cid, tau = job["config_id"], job["tau"]
    if cid not in cells:
        raise Malformed(f"unknown configuration {cid}")
    cell = cells[cid]
    first = tau if run in ACTIONABLE else 1
    domain_errs = []
    if job.get("first_date", 1) != first:
        domain_errs.append(f"first_date {job.get('first_date', 1)} != {first} for run {run}")
    prob = DecisionProblem(cell.dist, cell.T, cell.beta, tau, first_date=first)
    if "candidate_dates" in job and job["candidate_dates"] != prob.dates:
        domain_errs.append("stored candidate_dates differ from the run's domain")
    if job["T"] != cell.T:
        domain_errs.append(f"T {job['T']} != configuration's {cell.T}")
    D, W = prob.truth_argmax()
    if sorted(job["truth_argmax"]) != D:
        domain_errs.append(f"truth argmax {D} != stored {job['truth_argmax']}")
    if {int(k): parse(v) for k, v in job["truth_welfare"].items()} != W:
        domain_errs.append("stored exact truth welfare differs from recomputed")

    pair_errs, valid, seen = [], [], set()
    for p in job["pairs"]:
        key = (p.get("d"), p.get("r"))
        if key in seen:
            pair_errs.append(f"duplicate comparison {key}")
            continue
        seen.add(key)
        e = pair_structure(prob, D, p)
        if not e:
            e = check_pair(prob, cell, D, W, p)
        pair_errs += e
        if not e and p["status"] != "undecided":
            valid.append(p)

    outcome, date = derive_outcome(prob, D, valid)
    outcome_errs = []
    if outcome != job["outcome"]:
        outcome_errs.append(f"stored outcome {job['outcome']} not justified; "
                            f"verified comparisons give {outcome}")
    stored_date = job.get("identified_date")
    stored_date = None if stored_date in (None, "") else int(stored_date)
    if stored_date != date:
        outcome_errs.append(f"identified_date {stored_date} != justified {date}")
    return {"key": (cid, tau), "outcome": outcome, "date": date, "first": first,
            "domain_errors": domain_errs, "pair_errors": pair_errs,
            "outcome_errors": outcome_errs, "pairs": job["pairs"],
            "n_leaves": sum(len(p.get("certificate", [])) for p in job["pairs"]),
            "n_witness": sum(p.get("status") == "witness" for p in job["pairs"])}


def load_run(archive: Path, run: str):
    d = archive / "raw" / run
    if not d.is_dir():
        raise FileNotFoundError(f"archive for run {run!r} not found at {d}")
    jobs, errors = {}, []
    for f in sorted(d.iterdir()):
        m = NAME.match(f.name)
        if not m:
            errors.append(f"{run}: unexpected file {f.name}")
            continue
        key = (m.group(1), int(m.group(2)))
        try:
            with gzip.open(f, "rt", encoding="utf-8") as fh:
                job = json.load(fh)
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"{run}: {f.name} unreadable ({exc})")
            continue
        if not isinstance(job, dict) or (job.get("config_id"), job.get("tau")) != key:
            errors.append(f"{run}: {f.name} content does not match its file name")
            continue
        if key in jobs:
            errors.append(f"{run}: duplicate job {key}")
            continue
        jobs[key] = job
    return jobs, errors


def verify_run(archive: Path, run: str, cells, results: dict) -> list:
    t0 = time.time()
    jobs, errors = load_run(archive, run)
    src = RUNS[run]
    if src is None:
        expected = {(cid, tau) for cid in cells for tau in TAUS}
    else:
        if src not in results:
            raise RuntimeError(f"verify {src} before {run}")
        expected = {k for k, r in results[src].items() if r["outcome"] == "UNDECIDED"}
    missing, unknown = expected - set(jobs), set(jobs) - expected
    errors += [f"{run}: missing job {k}" for k in sorted(missing)]
    errors += [f"{run}: unexpected job {k}" for k in sorted(unknown)]

    out, n_leaves, n_wit = {}, 0, 0
    for key in sorted(set(jobs) & expected):
        try:
            res = verify_job(jobs[key], run, cells)
        except (Malformed, KeyError, ValueError, TypeError) as exc:
            errors.append(f"{run} {key}: malformed job ({exc})")
            continue
        if src is not None:
            base = {(p["d"], p["r"]): p for p in results[src][key]["pairs"]}
            for p in res["pairs"]:
                b = base.get((p["d"], p["r"]))
                if b is not None and b["status"] != "undecided" and b != p:
                    errors.append(f"{run} {key}: resolved comparison {(p['d'], p['r'])} "
                                  f"differs from {src}")
        for e in res["domain_errors"] + res["pair_errors"] + res["outcome_errors"]:
            errors.append(f"{run} {key}: {e}")
        n_leaves += res["n_leaves"]
        n_wit += res["n_witness"]
        out[key] = res
    results[run] = out
    print(f"{run}: {len(out)}/{len(expected)} jobs, {n_leaves} certificate leaves and "
          f"{n_wit} witnesses re-checked exactly; {len(errors)} errors "
          f"({time.time() - t0:.0f}s)", flush=True)
    return errors


def table2(results) -> tuple[list, list]:
    """Final classifications and the Table 2 counts, from verified outcomes."""
    errors, rows = [], []
    for panel, main, ext in (("all", "full", "extended"),
                             ("available", "actionable", "actionable_extended")):
        final = {k: dict(v, source=main) for k, v in results[main].items()}
        for k, v in results[ext].items():
            final[k] = dict(v, source=ext)
        for k in sorted(final):
            r = final[k]
            if r["outcome"] in ("UNDECIDED", "TIE_AT_TRUTH"):
                errors.append(f"Table 2 {panel}: {k} unresolved ({r['outcome']})")
            rows.append({"panel": panel, "config_id": k[0], "tau": k[1],
                         "outcome": r["outcome"], "identified_date": r["date"],
                         "source_run": r["source"]})
        expected = {(cid, tau) for cid in {k[0] for k in final} for tau in TAUS}
        if len(final) != 90 or set(final) != expected:
            errors.append(f"Table 2 {panel}: {len(final)} keys, expected 90")
    counts = []
    for tau in TAUS:
        a = [r for r in rows if r["panel"] == "all" and r["tau"] == tau]
        v = [r for r in rows if r["panel"] == "available" and r["tau"] == tau]
        cert_v = [r for r in v if r["outcome"] == "CERTIFIED_UNIQUE"]
        counts.append({
            "tau": tau,
            "best_date_identified": sum(r["outcome"] == "CERTIFIED_UNIQUE" for r in a),
            "best_date_ambiguous": sum(r["outcome"] == "NO_GUARANTEED_DATE" for r in a),
            "available_identified": len(cert_v),
            "act_now": sum(r["identified_date"] == tau for r in cert_v),
            "wait": sum(r["identified_date"] > tau for r in cert_v),
            "available_ambiguous": sum(r["outcome"] == "NO_GUARANTEED_DATE" for r in v),
        })
    ident = [r for r in rows if r["panel"] == "all" and r["outcome"] == "CERTIFIED_UNIQUE"]
    counts.append({"tau": "all", "identified_before_tau":
                   f"{sum(r['identified_date'] < r['tau'] for r in ident)}/{len(ident)}"})
    return rows, counts, errors


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="*", metavar="RUN",
                    help=f"runs to verify (default: all of {list(RUNS)})")
    ap.add_argument("--archive", default=str(DEFAULT_ARCHIVE))
    ap.add_argument("--out", default=None,
                    help="where to write table2_classifications.csv and the summary "
                         "(default: <archive>/verified)")
    a = ap.parse_args(argv)
    archive = Path(a.archive)
    bad = set(a.runs) - set(RUNS)
    if bad:
        ap.error(f"unknown run(s) {sorted(bad)}; choose from {list(RUNS)}")
    runs = [r for r in RUNS if r in (a.runs or RUNS)]
    for r in list(runs):                       # an extension needs its source verified
        if RUNS[r] and RUNS[r] not in runs:
            runs.insert(runs.index(r), RUNS[r])
    cells = cells_gamma1()
    results, errors = {}, []
    t0 = time.time()
    for run in runs:
        try:
            errors += verify_run(archive, run, cells, results)
        except FileNotFoundError as exc:
            errors.append(str(exc))
    if set(RUNS) <= set(results):
        rows, counts, e = table2(results)
        errors += e
        out = Path(a.out) if a.out else archive / "verified"
        out.mkdir(parents=True, exist_ok=True)
        import csv
        with open(out / "table2_classifications.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        (out / "table2_counts.json").write_text(json.dumps(counts, indent=2) + "\n")
        for c in counts:
            print("Table 2:", c)
    if WITNESS_NOTES:
        dev = max(abs(x[2]) for x in WITNESS_NOTES)
        print(f"note: {len(WITNESS_NOTES)} witnesses have an atomic part A with mass "
              f"!= 1 (max |mass-1| = {dev:.1e}); Q's unit mass and moments hold exactly")
    for e in errors:
        print("ERROR", e)
    print(f"{'PASSED' if not errors else f'FAILED with {len(errors)} errors'} "
          f"in {time.time() - t0:.0f}s")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
