# Verification of the final manuscript (2026-09-26)

Every quantitative claim in `paper/main.tex` was recomputed from the archived
outputs. `experiments/scripts/verify_manuscript_numbers.py` re-runs these checks
(all pass). This file records what was checked, how, and what was corrected.

## Consolidation

The final manuscript describes work that lived on four unmerged branches, all
forked from `main`. They were merged into this branch:

- `claude/window-a-atomic` (contains `claude/eloquent-brown-o9yxlz`): the
  corrected observation window, CMS Table 3, the robust-rule recomputation, and
  maximin.
- `claude/peaceful-archimedes-k8wa5b`: exact decision identification (Table 2).
- `claude/prediction-driven-allocation-eyllva`: an earlier atomic-truth study.
  Its seven conflicting files were resolved in favour of the corrected-window
  rerun on `window-a-atomic`.

## Results

| Item | How checked | Outcome |
|---|---|---|
| Table 1 (all 18 cells) | **New:** `extreme_pair_exact.py` reconstructs all 90 pairs with rational weights at the exact grid points i/2000, checks nonnegativity and exact moment matching, and computes welfare, optimal-date sets, spreads and ζ in rational arithmetic. | All match. 90/90 pairs are exact members of M_τ; every member has a unique optimum; disjoint sets 29/11/3. Before this, the repository had only floating-point LP solutions (residual ~1e-15), and "disagreement" was counted from tie-broken argmaxes, so the paper's claim of exact verification was not yet backed. |
| Table 2 (all cells), "all resolved", "27 of 44 precede τ" | Counts from `decision_identification/*_jobs.csv`; the exact verifier `verify_decision_certificates.py` was re-run on all four runs. | All match. 186/186 jobs re-verify: 54,070 certificate leaves and 87 witnesses. |
| Figure 2 caption (τ=2, optima 5 vs 1) | **New:** `fig2_indistinguishable_pair.py` builds the figure from the exact pair Beta(1,32), T=20, β=0.01. | Observation probabilities agree exactly; the optima are 5 and 1. This is the pair with the largest bound in Table 1 (ζ/2 = 6.34%). |
| Table 3 (all 42 cells), 1,169 = 589 + 580, 31 exclusions, CMS 1,200 decisions with 7 at 19 draws | `cms_table2_analyze.py` re-run, then independent recomputation | All match. The RC row equals `expected_regret_rule` on the shared ensemble for all 1,169 runs. `repro_reported_regret_calibrated.py` (new) reproduces all 7,147 archived bagging/RC decisions from cached trajectories and a sample from scratch. |
| Table 4, 525/597, Beta exclusions 75/3, 12/11/12 settings | Recomputed | All match. |
| Table 5, 7,189 runs, 11 empty sets, 324/801 and 22/7,189 | `robust_rerun_metrics.py` re-run; output identical to the archive | All match. |
| Inference (2,000 cluster replicates, seed 20260906, six-way Holm, 0/12 p10) | `final_inference_recompute.py` constants and output | All match. |
| Scan and grid settings (21/41 + 3×9; fine 201/201 + 7×41; 201-point grid; α=0.05; eigen threshold 1e-10) | Code | All match. CMS uses a 401-point grid, which the paper did not state; now added. |
| Lemma 1, exit-record equivalence, Prop. 1, Theorem 1, appendix derivations | 57/57 tests pass (exact rank 2τ for both statistics; window tests). Box bound, witness mass identity, MLR ordering, dual price range, and the τ(τ+3)/2 count were checked by hand. | Sound. |

## Manuscript corrections

**1. Observation window of the likelihood methods.** Six passages said that
NPMLE, EM, bagging and RC "additionally observe period-τ failures, revealing one
further moment". That described the superseded Window-B implementation. Every
number in the paper comes from the corrected code, in which all procedures use
the same pre-failure window, and both statistics span Π_{2τ−1}. The passages now
say this. They also say that the two families use different finite-sample
statistics (exit-record likelihood versus cohort moments). The affected
passages are: Section 3 (last sentence), Section 5.2 (Estimators), Section 6.2
(Information supplied), the Table 3 notes, Appendix A.2, Appendix C.1, the
Appendix D.1 text, the Conclusion, and the Reproducibility statement.

**2. Bootstrap retention figures (Appendix C.2).** These were Window-B counts
(1 skipped run, 6,792/407, 519 of 1,439,800, 7,199 ensembles). Table 3 uses the
Window-A run, for which the correct counts are:

- 53 runs skipped, 49 of them at τ=2;
- 4,913 runs retain all 200 draws, and 2,234 retain between 172 and 199;
- 9,502 of 1,429,400 fits are discarded (0.66%);
- 7,147 ensembles are archived.

The "31 exclusions" sentence (Section 5.3) now names their actual cause: the bagging
pipeline rejected the primary fit in those runs.

**Minor changes.** Table 3 column padding (7pt → 6pt) removes a 3 pt overfull
box. The CMS grid size was added.

## Also corrected in the repository

- `results/cms_table2_report.md` said 7 CMS runs returned no decision. The raw
  data show all 1,200 returned a decision, with 7 averaging over 19 draws.
- The tag `window-b-frozen`, cited by two reports, does not exist on the
  remote. Those reports now point to commit `197fd9e9`.

## Still author-held

The files `teaser4.png` (Figure 1) and `iclr2027_conference.bib` are not in the
repository; see `results/missing_submission_assets.md`. With placeholders for
both, the manuscript compiles without errors and the main text ends on page 8.
Check that the bib key `christensen2022optimal` refers to the CMS version used
in the experiments (RES 2026).

---

# Reconciliation with the 2026-09-30 manuscript PDF (2026-10-02)

The author's PDF of 2026-09-30 (19 pages) revises `paper/main.tex`. Its
wording was carried into `paper/main.tex`: tighter Related Work, Section 4 and
5 prose, the explicit definition of M_τ, citations for NPMLE, EM and bagging
(new keys `dempster1977maximum`, `breiman1996bagging`), the folded-in causal
scope sentence (A.4), the shortened A.5 and B.2 text, the revised Section 7.2
opening and Conclusion, and the new **Appendix D.3** with Tables 6 and 7. With
placeholders for Figure 1 and the bibliography, `main.tex` compiles without
errors to 19 pages, and the main text ends on page 8. A word-level diff of the
compiled text against the PDF leaves only the differences listed below.

## New experiment: Appendix D.3 (Tables 6 and 7)

The repository had no code or output for D.3. It was implemented from the
appendix's own specification in `experiments/scripts/matched_info_pairs.py`.
That script evaluates 10 procedures on both members of the 30 exact τ=2
pairs, with N ∈ {10³, 10⁵}, seeds 1001–1005, 40 bootstrap fits, and the
101-point LP grid with 9/11 scans and one 5-point refinement round. Its output
is `results/matched_info_pairs_raw.csv`, and
`matched_info_pairs_analyze.py` builds the tables from it. The run cost 0.28
core-hours. `verify_manuscript_numbers.py` now checks every D.3 number.

The PDF's D.3 numbers came from code that is not in the repository, so they
could not be reproduced exactly: the dataset draws differ. `main.tex` now
reports the reproducible run. Differences from the PDF:

| Quantity | PDF | Reproducible run |
|---|---|---|
| Retained trials (10³ / 10⁵) | 137 / 143 | 134 / 142 |
| Trials per pair | 3–5 | 2–5 |
| NPMLE invalid / Beta failures | 9 / 11 | 10 / 14 |
| Max change, date 1 on failure | 0.0021 | 0.0051 |
| Table 6, all cells | e.g. NPMLE 0.0648 / 0.0387 | e.g. NPMLE 0.0651 / 0.0375 |
| Table 7, EM55 at 10⁵ | −0.0048 [−0.0109, −0.0022] | −0.0029 [−0.0103, +0.00002] |
| Table 7, RC at 10³ | −0.0072 [−0.0141, −0.0009] | −0.0069 [−0.0213, +0.0040] |

The qualitative findings hold. EM55 has the lowest pooled pair risk at 10³,
bagging and RC tie for lowest at 10⁵, and EM55's point difference from NPMLE
is negative at both sample sizes. One claim does not hold: "intervals
excluding zero at both sample sizes". In the reproducible run the EM55
interval at 10⁵ reaches +1.8×10⁻⁵, and the D.3 results paragraph now says so.
If the author has the original D.3 code, adding it to the repository would
allow the PDF's numbers to be restored instead.

## Passages kept as verified, where the PDF differs

The PDF reverts the window and retention corrections made on 2026-09-26
(corrections 1 and 2 above). The archive contradicts the PDF's text, so
`main.tex` keeps the corrected statements:

- **Observation window.** The PDF says NPMLE, EM, bagging and RC "use exit
  records including period-τ failures, revealing one further moment". It says
  this in Section 3, Section 5.2, Section 6.2, the Table 3 notes, Appendix
  A.2, Appendix C.1, Appendix D.1 and the Reproducibility statement. The code
  behind every reported number uses the pre-failure window (Window A).
  `experiments/tests/test_observation_window.py` checks this, the run
  manifests record `observation_window: "A"`, and the PDF's own D.3 states
  that no procedure receives period-τ failures.
- **Appendix C.2 retention counts.** The PDF gives 1 skipped run, 6,792/407,
  519 of 1,439,800 (0.036%) and 7,199 ensembles. These are the superseded
  Window-B counts. The run behind Table 3 has 53 skipped runs (49 at τ=2),
  4,913/2,234, 9,502 of 1,429,400 (0.66%) and 7,147 ensembles, and
  `verify_manuscript_numbers.py` checks these against the archive.
- **Section 5.3 exclusions.** The PDF attributes the 31 exclusions to
  "missing bagging records". They are runs whose primary fit the bagging
  pipeline rejected.
- **Appendix C.3.** The CMS inner set uses a 401-point grid. The PDF omits
  this; `main.tex` keeps it.

The Table 3, Table 4 and Table 5 numbers in the PDF are identical to the
archive.

## Packaging

`experiments/scripts/package_supplement.py` builds a 60 MB anonymized
archive. The full repository is about 700 MB, while ICLR's supplementary
limit is 100 MB. Inside the unpacked archive the following were checked:
57/57 tests pass, `verify_manuscript_numbers.py` passes,
`cms_table2_analyze.py` scores 1,200 runs, `repro_reported_regret_calibrated.py`
reproduces all 1,169 Table 3 ensembles, and `matched_info_pairs_analyze.py`
reproduces Tables 6–7.

---

# Independent re-validation and corrections to the section above (2026-10-02, later)

The section above was re-checked independently. Its findings hold, with these
corrections and additions:

- **D.3 handling reversed.** The section above says `main.tex` "now reports
  the reproducible run". It no longer does. `main.tex` keeps the PDF's
  reported D.3 values, marked **unreproduced**
  (`release/paper_artifacts.yaml`). The reconstruction (`reconstruction-v1`)
  is a separate version, applied only through
  `paper/reconciliation/03_d3_reconstruction.patch`. A reported finding is
  not silently replaced.
- **"`verify_manuscript_numbers.py` now checks every D.3 number" overstated
  what it did.** That script compared archived files with constants written
  into the script and never read `paper/main.tex`. It is replaced by
  `release/verify_paper.py`, which reads the manuscript. The old `table2`
  "all 90 decided" check counted rows rather than outcomes. The Table 5
  refinement counts were printed rather than asserted. Tables 6–7 were
  compared as summary CSVs. All of this is fixed, and negative tests make
  each check fail on corrupted input.
- **Observation window, confirmed by recomputation, not by commit dates.**
  `experiments/audit/observation_window_audit.py`: 1,169/1,169 Table 3 NPMLE,
  1,200/1,200 EM55 and 1,122/1,122 Beta decisions equal Window-A
  recomputation. Under Window B, 654/1,168 and 1,138/1,200 agree. The
  Window-B archive serves as the control: 1,199/1,199 and 1,200/1,200 agree
  with recomputed Window B.
- **C.2 counts.** 53 / 49 / 7,147 / 4,913 / 2,234 (172–199) /
  9,502 of 1,429,400 recomputed from rows. The PDF's figures equal the
  Window-B archive exactly. The 31 Table 3 exclusions are the 11 + 20 τ=2
  skips at N = 10³ and 10⁵, a separate count from the full-grid skips.
- **New: C.4 archive was stale** (generated before the `nnls` correction).
  It is regenerated; the manuscript's 0/12 claim holds in both versions.
  See `paper/reconciliation/README.md` §04.
- **New: the provenance `source_dirty` flag never worked**, so the first D.3
  manifest claimed a clean tree. Fixed, and D.3 regenerated from a clean
  commit (identical except `runtime_s`).
- **New: certificate verification** now requires the expected job keys and
  re-derives each outcome from verified comparisons. All four archives pass:
  186 jobs, 54,070 leaves, 87 witnesses.
- **Assets:** Figure 1, Figure 2 (the authors' figure; the earlier generated
  `fig2.png` was a different rendering) and the bibliography are now present.

Full evidence: `release/evidence/REPORT.md` and `release/evidence/VALIDATION.md`.
