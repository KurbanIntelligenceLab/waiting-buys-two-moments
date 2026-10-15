# Decision identification — results

**Question.** Take the paper's population-level information after a prefix of
length τ. It is the set M_τ of all risk distributions on [0,1] whose moments
through degree 2τ−1 match the true population's. Is there a date that is
optimal for *every* distribution in M_τ?

**Scope.** 30 configurations (the γ = 1 frozen grid: 5 distributions × T ∈ {12, 20}
× β ∈ {0.0005, 0.002, 0.01}) at τ = 2, 3, 4, for 90 configuration–prefix pairs.
Candidate dates are s ∈ {1, …, T−1}, matching `PopulationConfig.dates` and the
paper's t\*. The simulator, estimators, existing benchmarks and manuscript are
unchanged.

**Method.** `REDUCTION.md` gives the reduction and its proofs, written before any
computation.
- For each pair (configuration, τ), the true population's best date d is
  computed exactly.
- For each competing date r, the largest advantage V(d, r) that r can have over
  d anywhere in M_τ is then either:
  - **certified negative**: every sub-box of the allocation/price parameters
    carries a dual certificate whose nonnegativity is checked exactly on all of
    [0,1], not on a grid; or
  - **shown positive**: an exact witness Q ∈ M_τ with W^r(Q) > W^d(Q). Its
    moments are checked equal in rational arithmetic, and its welfare is
    computed exactly.
- Floating-point LPs only propose multipliers and candidate witnesses. Every
  conclusion below rests on exact rational checks.

## Headline

| τ | moments identified | **certified: unique optimal date** | **refuted: no date optimal for all of M_τ** | undecided |
|---|---|---|---|---|
| 2 | 3 | **1 / 30** | **29 / 30** | 0 |
| 3 | 5 | **16 / 30** (14 main + 2 extended) | **14 / 30** | 0 |
| 4 | 7 | **27 / 30** (26 main + 1 extended) | **3 / 30** | 0 |
| all | | 44 | 46 | **0** |

- The main run, with a fixed budget of 3000 boxes or 900 s per pair, decided
  87 of 90 pairs: 41 certified, 46 refuted, 3 undecided.
- The 3 undecided pairs were then re-run alone with a larger budget (30,000
  boxes or 2 h per pair). All 3 were certified. See "Extended pass" below; the
  main-run outcome is kept on file unchanged.

**The three decision targets.**
- The exact argmax of the true population is a single date in all 90 pairs, so
  there is no tie at the truth.
- Every decided case is decided strictly.
- A certified configuration therefore has a date that is simultaneously
  *guaranteed optimal*, *uniquely optimal* and the *identified earliest optimal
  date* under the paper's earliest-date tie-breaking.
- A refuted configuration has none of the three (REDUCTION.md, Prop. 1(d–e)).
- The three targets could differ only through exact ties across all of M_τ. The
  method cannot certify such a tie from above. None arose among the decided
  cases.

**Every case on this grid is resolved.** After the extended pass, each of the
90 configuration–prefix pairs is decided, and each decision is backed by an
exact certificate or an exact witness. There are no undecided cases and no
ties at the truth. This is a complete resolution of *these 90 cases* in the
paper's model. It is not a general characterization of when the optimal date
is identified for arbitrary populations: no closed-form condition is claimed.

## Comparison with the paper's Table 5 ("unique optimal date, upper bound")

| τ | Table 5 upper bound | certified here | refuted here | exact fraction (main run alone) | **exact fraction (with extended pass)** |
|---|---|---|---|---|---|
| 2 | 1/30 (3.3%) | 1 | 29 | exactly 3.3% | **3.3%** |
| 3 | 18/30 (60.0%) | 16 | 14 | in [46.7%, 53.3%] | **16/30 = 53.3%** |
| 4 | 27/30 (90.0%) | 27 | 3 | in [86.7%, 90.0%] | **27/30 = 90.0%** |

Table 5's upper bounds turn out to be exact at τ = 2 and τ = 4, and 2
configurations too high at τ = 3.

- **Consistent with every earlier witness.** All 44 pairs that the
  reachable-date search or the extreme-pair LP had already shown non-unique are
  refuted here too, with this method's own exact witnesses. None of them was
  certified.
- **New refutations.** Two τ = 3 pairs that Table 5 counted as possibly unique
  are refuted by new exact witnesses:

  | Configuration | Witness |
  |---|---|
  | `eeb7694df15d`, Beta(1,4), T=12, β=0.01 | date 3 beats date 2 by 0.062% of W\* |
  | `1bb4f8059848`, Beta(1,4), T=20, β=0.0005 | date 5 beats date 4 by 0.011% of W\* |

  Both advantages are small but exactly positive. They correct the τ = 3
  upper bound from 18/30 to at most 16/30.
- **Open cases resolved.** Of Table 5's 46 open pairs (1 + 18 + 27), 44 are
  certified unique (41 in the main run and 3 in the extended pass) and 2 are
  refuted.

## Per-configuration results

✓ = certified unique optimal date (bold). ✗ (d vs r) = refuted: an exact member
of M_τ where date r beats the truth's optimum d. ✓ ext. = undecided in the main
run and certified in the extended pass. *new* = previously counted as possibly
unique.

| Distribution | T | β | config | τ=2 | τ=3 | τ=4 |
|---|---|---|---|---|---|---|
| beta_1_32 | 12 | 0.0005 | `08d96ab0defb` | ✗ (2 vs 4) | **2** ✓ | **2** ✓ |
| beta_1_32 | 12 | 0.002 | `c8697d1800b5` | ✗ (2 vs 1) | **2** ✓ | **2** ✓ |
| beta_1_32 | 12 | 0.01 | `9fe557d42d19` | **1** ✓ | **1** ✓ | **1** ✓ |
| beta_1_32 | 20 | 0.0005 | `82d22f34a4dc` | ✗ (4 vs 2) | **4** ✓ ext. | **4** ✓ |
| beta_1_32 | 20 | 0.002 | `82fa2d570256` | ✗ (3 vs 2) | ✗ (3 vs 2) | **3** ✓ |
| beta_1_32 | 20 | 0.01 | `fad987cc30ad` | ✗ (5 vs 4) | ✗ (5 vs 4) | **5** ✓ |
| beta_1_8 | 12 | 0.0005 | `645ba2bddb66` | ✗ (4 vs 3) | ✗ (4 vs 3) | **4** ✓ |
| beta_1_8 | 12 | 0.002 | `48992d9e3b43` | ✗ (3 vs 2) | **3** ✓ | **3** ✓ |
| beta_1_8 | 12 | 0.01 | `9d58c5dce3c9` | ✗ (2 vs 1) | **2** ✓ | **2** ✓ |
| beta_1_8 | 20 | 0.0005 | `fae85d196c89` | ✗ (4 vs 3) | **4** ✓ | **4** ✓ |
| beta_1_8 | 20 | 0.002 | `51c48458632c` | ✗ (3 vs 2) | **3** ✓ | **3** ✓ |
| beta_1_8 | 20 | 0.01 | `fdeab5019bb9` | ✗ (2 vs 5) | **2** ✓ | **2** ✓ |
| beta_1_4 | 12 | 0.0005 | `6ee1a95c2835` | ✗ (4 vs 3) | ✗ (4 vs 3) | **4** ✓ |
| beta_1_4 | 12 | 0.002 | `d157b5b6b9b0` | ✗ (3 vs 4) | ✗ (3 vs 4) | **3** ✓ |
| beta_1_4 | 12 | 0.01 | `eeb7694df15d` | ✗ (2 vs 3) | ✗ (2 vs 3) *new* | ✗ (2 vs 3) |
| beta_1_4 | 20 | 0.0005 | `1bb4f8059848` | ✗ (4 vs 5) | ✗ (4 vs 5) *new* | **4** ✓ ext. |
| beta_1_4 | 20 | 0.002 | `89a9f7e78122` | ✗ (4 vs 3) | ✗ (4 vs 3) | **4** ✓ |
| beta_1_4 | 20 | 0.01 | `9cc5ff7a0659` | ✗ (3 vs 2) | ✗ (3 vs 2) | **3** ✓ |
| mixture_high_risk_subgroup | 12 | 0.0005 | `76324efe4052` | ✗ (3 vs 2) | **3** ✓ | **3** ✓ |
| mixture_high_risk_subgroup | 12 | 0.002 | `d3ad67d001b3` | ✗ (2 vs 3) | **2** ✓ | **2** ✓ |
| mixture_high_risk_subgroup | 12 | 0.01 | `6eb1e1952502` | ✗ (1 vs 2) | ✗ (1 vs 2) | **1** ✓ |
| mixture_high_risk_subgroup | 20 | 0.0005 | `6a778be2d26d` | ✗ (3 vs 2) | **3** ✓ | **3** ✓ |
| mixture_high_risk_subgroup | 20 | 0.002 | `f3e14f3cb88f` | ✗ (2 vs 3) | **2** ✓ | **2** ✓ |
| mixture_high_risk_subgroup | 20 | 0.01 | `7ee9ad48135e` | ✗ (2 vs 1) | ✗ (2 vs 1) | ✗ (2 vs 1) |
| bimodal | 12 | 0.0005 | `7a44ab9f6ee6` | ✗ (3 vs 4) | ✗ (3 vs 4) | **3** ✓ |
| bimodal | 12 | 0.002 | `a22714b128d1` | ✗ (3 vs 2) | **3** ✓ | **3** ✓ |
| bimodal | 12 | 0.01 | `0f904f38c8bf` | ✗ (2 vs 1) | **2** ✓ | **2** ✓ |
| bimodal | 20 | 0.0005 | `06380b49a6f3` | ✗ (4 vs 3) | **4** ✓ ext. | **4** ✓ |
| bimodal | 20 | 0.002 | `67ddeab9efe8` | ✗ (3 vs 4) | ✗ (3 vs 4) | ✗ (3 vs 4) |
| bimodal | 20 | 0.01 | `25798ac797cd` | ✗ (4 vs 3) | ✗ (4 vs 3) | **4** ✓ |

## How large are the margins?

- **Certified pairs.** The certified upper bound on any competitor's advantage,
  relative to W\*(P), has median −1.0×10⁻⁴. It ranges from −4.6×10⁻³ to
  −6.4×10⁻⁷. These are *upper* bounds: the true worst case can be more negative.
  Even so, many decisions are identified by a thin margin: some other population
  in M_τ comes close to making a later or earlier date optimal.
- **Refuted pairs.** The exact witness advantage has median 0.45% of W\*,
  maximum 6.4% (`c8697d1800b5`, Beta(1,32), T=12, β=0.002, at τ = 2), and
  minimum 4.2×10⁻⁵ (`eeb7694df15d` at τ = 4).
  - Witnesses are built by perturbing an LP solution toward the true
    population. Their advantages are therefore *lower* bounds on V(d, r), not
    the worst case.
  - Theorem 1's ζ/2 regret floors come from the extreme-pair construction, not
    from these witnesses.

## Cases undecided in the main run

In each undecided pair, the box LP could not push the upper bound below zero
within the per-pair budget (3000 boxes or 900 s), and no witness was found. The
remaining float upper bound on the competitor's advantage was below 10⁻³ of β
in every case, which is a numerical near-tie. Following the protocol, they
were left undecided in the main run.

| config | τ | truth optimum d | open competitor r | r's gap at the truth (rel.) | open float upper bound / β |
|---|---|---|---|---|---|
| `82d22f34a4dc` Beta(1,32), T=20, β=0.0005 | 3 | 4 | 5 | −1.04% | 1.6×10⁻⁴ |
| `06380b49a6f3` bimodal, T=20, β=0.0005 | 3 | 4 | 5 | −0.12% | 2.5×10⁻⁴ |
| `1bb4f8059848` Beta(1,4), T=20, β=0.0005 | 4 | 4 | 5 | −0.074% | 6×10⁻⁶ |

In all three, every other competitor is certified. The only question left in
each is whether date 5 can beat date 4 somewhere in M_τ.

The main run spent its full 900 s on each of these pairs, and the time limit,
not the box limit, is what stopped them.

## Extended pass on the undecided pairs

The three pairs above were re-run alone, with the same code and the same exact
checks. The budget was raised to 30,000 boxes or 2 h per pair. Every other pair
in those three jobs keeps its main-run result.

| config | τ | pair (d vs r) | outcome | boxes | certificate leaves | time | certified bound on r's advantage (rel. to W\*) |
|---|---|---|---|---|---|---|---|
| `82d22f34a4dc` | 3 | 4 vs 5 | **certified** | 1,590 | 798 | 990 s | < −6.4×10⁻⁷ |
| `06380b49a6f3` | 3 | 4 vs 5 | **certified** | 1,098 | 552 | 1,512 s | < −3.1×10⁻⁷ |
| `1bb4f8059848` | 4 | 4 vs 5 | **certified** | 476 | 241 | 655 s | < −1.5×10⁻⁶ |

- All three jobs become CERTIFIED_UNIQUE with date 4.
- Their 4,367 certificate leaves were re-verified exactly from disk
  (`extended_verification.txt`).
- The certified margins are tiny. Date 5 can come within about one part in a
  million of date 4's welfare, and that is why these pairs were slow. They are
  genuine near-ties that turned out to be decidable. They are not exact ties.

## An interpretive point: identified is not the same as still actionable

The candidate set is the paper's {1, …, T−1}, which includes dates earlier than
the prefix τ. Of the 44 certified dates (main run plus extended pass), 27 are
*before* τ, 13 equal τ, and 4 are after τ.

| τ | certified date < τ | = τ | > τ |
|---|---|---|---|
| 2 | 1 | 0 | 0 |
| 3 | 8 | 5 | 3 |
| 4 | 18 | 8 | 1 |

So in most certified cases, the observations settle the best date only after
that date has passed. A planner who waits for τ periods learns that it should
have acted earlier. The follow-up below asks the companion question: whether
the best date *still available*, s ≥ τ, is identified.

## Follow-up: restricting to dates still available (s ≥ τ)

The same experiment was re-run with candidate dates {τ, …, T−1}. It uses the
same code, the same exact checks, the same budgets, and the same extended pass
on undecided pairs. Only the candidate set changes; REDUCTION.md, Prop. 1 holds
for any candidate set.

| τ | **certified: best available date** | of which "act now" (d = τ) | of which "wait" (d > τ) | **refuted** | undecided |
|---|---|---|---|---|---|
| 2 | **3 / 30** | 3 | 0 | **27 / 30** | 0 |
| 3 | **18 / 30** (16 main + 2 extended) | 14 | 4 | **12 / 30** | 0 |
| 4 | **28 / 30** (27 main + 1 extended) | 27 | 1 | **2 / 30** | 0 |
| all | 49 | **44** | **5** | 41 | **0** |

**Every one of the 90 pairs is decided here as well.** The main run left the
same three near-ties undecided (date 4 vs 5). Their extended pass certified
all three and reproduced the main extended pass exactly, with the same box
counts and bounds.

**What this shows.**
- Restricting to available dates does *not* make the decision rarely
  identified. It is identified in 49 of 90 pairs, against 44 of 90 without the
  restriction.
- What changes is the answer. In **44 of the 49** identified cases, the
  identified date is τ itself: the observations determine that the planner
  should act immediately. Only **5** certify that waiting further is optimal:
  - four at τ = 3 (date 4): `08d96ab0defb`, `82d22f34a4dc`, `fae85d196c89`,
    `06380b49a6f3`;
  - one at τ = 4 (date 5): `fad987cc30ad`.
- In the 34 pairs whose unrestricted optimum had already passed, the best
  available date is τ in 32. It is certified in 28 and refuted in 6.
- So, on this grid, waiting τ periods usually buys enough information to decide
  one thing: stop waiting. The observations settle "act now" far more often
  than they settle "wait longer".
- Refuted pairs have a median witness advantage of 0.44% of W\* (max 6.5%).
  In 28 of the 41, the competitor that beats the truth's best available date
  is a *later* date. There, the observations cannot rule out that waiting
  longer would be better.

**Checks.**
- All 90 jobs plus the 3 extended ones were re-verified exactly from disk:
  25,554 certificate leaves and 41 witnesses (`actionable_verification.txt`,
  `actionable_extended_verification.txt`).
- The results were checked for logical consistency against the unrestricted
  run. In the 56 pairs where the unrestricted optimum is ≥ τ, the optimum
  agrees. Every certificate that should carry over does, and no pair (d, r) is
  certified in one run and refuted in the other. There were 0 contradictions.

**Runtime.** The main pass took 1 h 35 min on 4 workers (22,046 CPU-s). The
extended pass took 24 min on 3 workers (2,927 CPU-s).

**Files.** `raw/actionable/`, `raw/actionable_extended/`, `actionable_*.csv`
and `actionable_extended_*.csv`. `actionable_final_jobs.csv` merges the two,
with the unrestricted outcome alongside for comparison. In the actionable CSVs,
the `prior_status` column still refers to the *unrestricted* Table 5 status and
has no meaning for this candidate set.

## Validation performed before the full run

1. **Unit tests** (`experiments/tests/test_decision_id.py`, 9/9 pass). Each step
   of the reduction is checked in exact arithmetic on random populations:
   - group ordering;
   - threshold allocations = greedy fill, including empty groups and a budget
     that exceeds the whole pool;
   - dual form with price in [0,1];
   - the Bernstein bound is rigorous, including a dip narrower than a
     1001-point grid;
   - certificates dominate G on sampled members of M_τ;
   - corrupted multipliers never undercut an attained value;
   - witnesses are exact members of M_τ;
   - the exact truth argmax matches the repository's t\* on all 30 cells.
2. **Pilot** (9 jobs).
   - The four configurations already known to be ambiguous are refuted by exact
     witnesses. These are Theorem 1's case and the three search false-positives
     at τ = 2, 3, 4.
   - The five open configurations (the τ = 2 one and four τ = 4 ones across
     families) are certified.
   - All certificates were re-verified from disk.
   - The first pilot exposed a HiGHS presolve failure on refined grids. It cost
     only completeness, never soundness. It was fixed by retrying without
     presolve, and the pilot was re-run on the fixed code. Only that version's
     results are kept.
3. **Soundness check.** With witness search disabled on the four ambiguous
   configurations, the certifier certified none of the witnessed pairs. Each
   stayed undecided. Pairs it did certify there are consistent with every known
   witness.
4. **Independent re-verification of the full run**
   (`verify_decision_certificates.py`, output in `full_verification.txt`).
   - All 90 jobs pass.
   - 24,149 certificate leaves were re-checked exactly, and the leaves of every
     certified pair tile the parameter domain exactly (checked for area and
     disjointness).
   - For each of the 46 witnesses, every moment was recomputed exactly in the
     monomial basis, 1 + g ≥ 0 was confirmed on [0,1], and W^r > W^d was
     confirmed exactly.

## Runtime

| Stage | Wall time | CPU |
|---|---|---|
| Pilot (fixed code) | 10 min | ≈ 25 min |
| Soundness check | 8 min | ≈ 15 min |
| Full run, 90 jobs on 4 workers | 1 h 38 min | 6.3 h (22,816 s) |
| Re-verification of all certificates and witnesses | 51 s | 51 s |
| Extended pass (3 pairs, 3 workers) | 25 min | 53 min (3,157 s) |

By outcome:
- **Certified jobs:** 12,370 CPU-s in total, median 198 s per job.
- **Refuted jobs:** median 0.7 s. A witness is usually found on the first boxes.
- **Undecided jobs:** 5,220 CPU-s, spent mostly at the 900 s per-pair limit.

Across the run there were 49,430 box LPs.

## Limitations

- **Complete for this grid, not in general.** All 90 pairs are decided, but
  3 needed the extended budget. The method certifies strict inequalities only.
  If a configuration had an exact tie across all of M_τ (V(d, r) = 0), it could
  never be certified; it could only be shown by an exact tie witness. So on
  other configurations the method can leave cases undecided. On this grid no
  such tie arose, and |D\*(P)| = 1 in all 90.
- **Model scope.** These are mean-field, population-level results at γ = 1 with
  the paper's information set (moments through degree 2τ−1). They say nothing
  about finite samples. Nor do they cover the individual-exit-record
  convention, which the code's estimators currently use and which carries one
  more moment (see the earlier discussion).
- **Candidate dates.** The headline results use the paper's {1, …, T−1},
  which includes past dates. The follow-up section repeats the whole
  experiment with the available dates {τ, …, T−1}.
- **Budgets.** The per-pair budget (3000 boxes, 900 s) was fixed after the pilot
  and before the full run, and was the same for every pair. "Undecided" means
  undecided at that budget.
- **Margins.** Certified bounds are upper bounds on the worst-case advantage,
  and witness advantages are lower bounds on it. Neither is the exact maximum
  regret.
- **What the conclusions rest on.** They rest on the proofs in REDUCTION.md and
  on exact rational arithmetic (Python `fractions`). The proofs have not had an
  independent human review. They should be checked before any of this goes into
  the paper.

## Files

| Path | Contents |
|---|---|
| `REDUCTION.md` | the reduction, proofs, and classification rules |
| `wtm/decision_id.py` | certifier: box LPs, exact Bernstein check, exact witnesses, branch and bound |
| `experiments/scripts/decision_identification.py` | driver (`pilot`, `soundness`, `full`, `extend`) |
| `experiments/scripts/verify_decision_certificates.py` | independent exact re-verification from disk |
| `experiments/scripts/decision_identification_summary.py` | CSV tables from raw outputs |
| `experiments/tests/test_decision_id.py` | tests of each lemma and of certificate soundness |
| `raw/full/*.json.gz` | one file per job: exact truth welfare, per-pair results, every certificate leaf (box, multipliers, U_cert) and every witness (atoms, weights, ε, g, exact welfare) |
| `full_jobs.csv`, `full_pairs.csv` | flat tables |
| `full_verification.txt` | re-verification log |
| `raw/pilot2/`, `raw/soundness/`, `pilot2_*.csv`, `soundness_verification.txt` | pilot and soundness check |
| `raw/extended/`, `extended_*.csv` | extended pass on undecided pairs |
| `raw/actionable/`, `raw/actionable_extended/`, `actionable_*.csv`, `actionable_final_jobs.csv` | follow-up with candidate dates s ≥ τ |
