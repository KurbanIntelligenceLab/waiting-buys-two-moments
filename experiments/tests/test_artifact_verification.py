"""Negative checks for saved trials and decision-certificate archives.

Manuscript-specific verification is maintained on the main branch.
"""
from __future__ import annotations
import gzip
import json
import shutil
import sys
from pathlib import Path
import pandas as pd
import pytest
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.scripts import matched_info_pairs_analyze as mipa
from experiments.scripts import verify_decision_certificates as vdc
RESULTS = ROOT / "results"
DI = RESULTS / "decision_identification"


def test_duplicate_d3_trial_fails():
    raw = pd.read_csv(RESULTS / "matched_info_pairs_raw.csv")
    with pytest.raises(mipa.RawError):
        mipa.check_raw(pd.concat([raw, raw.iloc[[0]]]))


def test_missing_d3_procedure_fails():
    raw = pd.read_csv(RESULTS / "matched_info_pairs_raw.csv")
    with pytest.raises(mipa.RawError):
        mipa.check_raw(raw[raw.procedure != "em_80"])


# -- certificate verifier -----------------------------------------------------
@pytest.fixture
def fast_pairs(monkeypatch):
    """Skip the exact certificate/witness arithmetic; keep every structural check
    (status, domain, reference date, coverage, outcome justification).

    The exact checks are covered by the full run of
    verify_decision_certificates.py. These tests target coverage, domain and
    outcome justification.
    """
    monkeypatch.setattr(vdc, "check_pair", lambda prob, cell, D, W, p: [])


def archive_copy(tmp: Path) -> Path:
    dst = tmp / "di"
    (dst / "raw").mkdir(parents=True)
    for run in vdc.RUNS:
        shutil.copytree(DI / "raw" / run, dst / "raw" / run)
    return dst


def edit_job(path: Path, fn):
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        job = json.load(fh)
    fn(job)
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        json.dump(job, fh)


def first_job(arch, run, outcome):
    for f in sorted((arch / "raw" / run).iterdir()):
        with gzip.open(f, "rt", encoding="utf-8") as fh:
            if json.load(fh)["outcome"] == outcome:
                return f
    raise AssertionError(outcome)


def verify_archive(arch, tmp):
    return vdc.main(["--archive", str(arch), "--out", str(tmp / "out")])


def test_certificate_baseline_passes(tmp_path, fast_pairs):
    assert verify_archive(archive_copy(tmp_path), tmp_path) == 0


def test_certificate_absent_run_fails(tmp_path, fast_pairs):
    arch = archive_copy(tmp_path)
    shutil.rmtree(arch / "raw" / "actionable_extended")
    assert verify_archive(arch, tmp_path) == 1


def test_certificate_missing_job_fails(tmp_path, fast_pairs):
    arch = archive_copy(tmp_path)
    sorted((arch / "raw" / "full").iterdir())[0].unlink()
    assert verify_archive(arch, tmp_path) == 1


def test_certificate_duplicate_job_fails(tmp_path, fast_pairs):
    arch = archive_copy(tmp_path)
    f = sorted((arch / "raw" / "full").iterdir())[0]
    shutil.copy(f, f.with_name("000000000000_tau2.json.gz"))
    assert verify_archive(arch, tmp_path) == 1


def test_certificate_unjustified_unique_fails(tmp_path, fast_pairs):
    arch = archive_copy(tmp_path)
    f = first_job(arch, "full", "CERTIFIED_UNIQUE")
    edit_job(f, lambda j: j["pairs"].pop())        # drop one competitor's certificate
    assert verify_archive(arch, tmp_path) == 1


def test_certificate_witness_outside_domain_fails(tmp_path, fast_pairs):
    arch = archive_copy(tmp_path)
    f = first_job(arch, "actionable", "NO_GUARANTEED_DATE")

    def move(j):
        for p in j["pairs"]:
            if p["status"] == "witness":
                p["r"] = 1                          # before tau: not an available date
    edit_job(f, move)
    assert verify_archive(arch, tmp_path) == 1


def test_certificate_unknown_status_fails(tmp_path, fast_pairs):
    arch = archive_copy(tmp_path)
    f = first_job(arch, "full", "CERTIFIED_UNIQUE")
    edit_job(f, lambda j: j["pairs"][0].update(status="probably_fine"))
    assert verify_archive(arch, tmp_path) == 1


def test_certificate_undecided_extension_fails(tmp_path, fast_pairs):
    arch = archive_copy(tmp_path)
    f = sorted((arch / "raw" / "extended").iterdir())[0]

    def undecide(j):
        j["outcome"], j["identified_date"] = "UNDECIDED", None
        for p in j["pairs"]:
            if p.get("extended_from_main_run"):
                p["status"] = "undecided"
    edit_job(f, undecide)
    assert verify_archive(arch, tmp_path) == 1
