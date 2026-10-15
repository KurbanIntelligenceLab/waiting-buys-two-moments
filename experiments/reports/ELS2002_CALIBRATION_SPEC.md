# ELS:2002 calibration specification (Experiments 25 and 26)

**Frozen 2026-08-30, before any rule was evaluated on this cohort.** Every
choice below is made from the NCES codebook definitions alone. No rule
performance was computed, viewed, or used to select any variable, filter or
checkpoint.

## Dataset

| Item | Value |
| --- | --- |
| Study | Education Longitudinal Study of 2002 (ELS:2002) |
| File | `els_02_12_byf3pststu_v1_0.sav` (SPSS, 87.97 MB) |
| Obtained from | `https://nces.ed.gov/EDAT/Data/Zip/ELS_2002-12_PETS_v1_0_Student_SPSS_Datasets.zip` |
| Zip size / date | 18,448,509 bytes; file dated 2015-04-01 |
| Access class | **Public-use.** Direct HTTP download, no licence, no interactive codebook |
| Downloaded | 2026-08-30 |
| Records | 16,197 sample members; 4,012 variables |

Chosen because it is a second NCES longitudinal education cohort with dropout
explicitly tracked, independent of NELS:88, and public-use.

## Cohort and time

| Item | Value |
| --- | --- |
| Cohort | ELS:2002 base-year **spring-2002 10th-grade cohort** |
| Time origin | Base year, spring 2002 |
| Checkpoint 1 -> model period 1 | First follow-up, **spring 2004** |
| Checkpoint 2 -> model period 2 | Second follow-up, **2006** |
| Held out, not used in the fit | Third follow-up, 2012 |

**Wave-to-period mapping.** Model periods are abstract and equally spaced. BY
(2002) -> F1 (2004) -> F2 (2006) are equally spaced at **two calendar years
each**, so mapping wave index to model period introduces no distortion across
the two checkpoints used for calibration. The third follow-up (2012) is six
years after F2 and is therefore **excluded from the model-period mapping
entirely**; it is retained only as an out-of-fit descriptive check in
Experiment 26. One model period means two calendar years here, and that is
stated rather than assumed.

## Failure definition

Failure = **a recorded high-school dropout episode.** Alternative completers
(early GED recipients with no dropout episode) are **not** counted as failures,
matching the source paper's treatment of dropout as the failure event.

| Quantity | Variable | Codes counted as failed |
| --- | --- | --- |
| `c1`, cumulative dropout by spring 2004 | `F2SP04DO` (Spring 2004 dropout status, updated) | `1` F1-identified spring-2004 dropout; `3` F2-identified spring-2004 dropout |
| `c2`, cumulative dropout by 2006 | `F2DOSTAT` (F2 dropout status as of 2006 interview) | `1` dropout at F2; `2` HS student with prior dropout episode; `4` HS graduate with prior dropout episode; `5` dropout episode, F2 status unknown |

`F2SP04DO` is preferred over `F1DOSTAT` because it is the *updated* spring-2004
status, incorporating dropouts identified retrospectively at F2; using the
un-updated F1 variable would undercount `c1` and bias the fit. Both checkpoints
count dropout *episodes*, so `c1` and `c2` are consistent with each other and
`c2 >= c1` by construction.

Codes excluded as not-failed at both checkpoints: early GED / alternative
completers (`F2SP04DO` in {2, 4}; `F2DOSTAT` = 3).

## Missing data

Listwise on the two calibration variables: a sample member is retained only if
`F2SP04DO != -9` and `F2DOSTAT` is non-missing. No imputation. The number
dropped is reported. Out-of-scope members are excluded by the same rule.

## Survey weights

**Weighted**, using `F2BYWT` (second follow-up base-year panel weight) -- the
weight designed for a base-year cohort followed through 2006, which is exactly
the span of the two checkpoints. Members with zero or missing weight are
excluded. Unweighted proportions are computed and reported alongside as a
sensitivity check; the **weighted** figures are the frozen calibration.

## Calibration family and method

Beta, fitted by moment inversion, for direct comparability with the published
NELS calibration.

Under our model at `gamma = 1` with absorbing failure and per-period hazard `p`:

    S1 = E[1-p]      = 1 - E[p]                  =>  c1 = E[p]
    S2 = E[(1-p)^2]  = 1 - 2E[p] + E[p^2]        =>  c2 = 2E[p] - E[p^2]

so the two observed cumulative dropout fractions identify the first two moments
exactly:

    m0 = E[p]   = c1
    m1 = E[p^2] = 2*c1 - c2

and the Beta parameters follow from the same inversion the source paper uses:

    s = (m0 - m1) / (m1 - m0^2),   alpha = m0 * s,   beta = s - alpha

**Deviation from the source, recorded.** Shirali et al. describe `m1` verbally
as "the proportion of the initial pool who failed right before the second step",
while their stated limit for it is `E[p^2]`. Under an absorbing-failure reading
those are not the same quantity (that proportion is `E[p] - E[p^2]`). The
ambiguity is documented in `NELS_PROVENANCE.md`. We resolve it **in favour of
their stated limits**, deriving `m1` from survival as above, because that is
internally consistent with our own generative model. Applying this mapping to
their published pair implies NELS cumulative dropout of `c1 = 7.41%` then
`c2 = 9.29%`, which is the reasonableness check on the resolution.

## Validity conditions, checked before use

1. `0 <= c1 <= c2 <= 1`
2. `m1 >= m0^2` (variance non-negative) -- else the two-moment Beta fit does not exist
3. `m1 <= m0` (Hausdorff: `E[p(1-p)] >= 0`)
4. `alpha > 0`, `beta > 0`
5. Round-trip: re-deriving `(m0, m1)` from the fitted `(alpha, beta)` returns the inputs

If any of 1-4 fails, the Beta calibration is reported as **non-existent for this
cohort** rather than repaired, and that is itself the finding.

## Preserved for Experiment 26

The full empirical longitudinal attrition record -- dropout counts and rates at
every available checkpoint including the held-out 2012 wave, weighted and
unweighted, with denominators -- is written to
`results/els2002_attrition_record.csv` regardless of whether the Beta fit
succeeds, so that Experiment 26 can ask whether Beta describes the cohort
without re-reading the raw file.

---

# Exact algebra: raw statuses -> moments -> Beta parameters

Added 2026-08-30 at the author's instruction, so the whole chain is auditable
without re-running anything. Every number below is reproduced by
`python -m experiments.scripts.els2002_calibrate`.

## Step 0 -- sample selection

    total sample members                                   16,197
    dropped by the frozen missing/weight rule               2,189
      (F2SP04DO == -9, or F2DOSTAT missing,
       or F2BYWT missing or <= 0)
    retained                                               14,008
    sum of F2BYWT over retained                       3,354,119.3

## Step 1 -- raw status counts among the retained sample

`F2SP04DO`, spring-2004 dropout status (updated):

| code | meaning | n | sum of weights | counted as failed |
| ---: | --- | ---: | ---: | :---: |
| 0 | not spring-04 dropout / early GED | 13,101 | 3,076,839.0 | |
| 1 | F1-identified spring-2004 dropout | 580 | 179,390.2 | **yes** |
| 2 | F1-identified early GED recipient | 118 | 31,649.9 | |
| 3 | F2-identified spring-2004 dropout | 182 | 58,057.8 | **yes** |
| 4 | F2-identified early GED recipient | 27 | 8,182.5 | |

`F2DOSTAT`, dropout status as of the 2006 interview:

| code | meaning | n | sum of weights | counted as failed |
| ---: | --- | ---: | ---: | :---: |
| 0 | no known dropout episode | 12,478 | 2,897,640.2 | |
| 1 | dropout at time of F2 interview | 726 | 227,088.9 | **yes** |
| 2 | HS student in F2, prior dropout episode | 45 | 14,887.4 | **yes** |
| 3 | GED recipient | 458 | 130,900.8 | |
| 4 | HS graduate with prior dropout episode | 300 | 83,502.1 | **yes** |
| 5 | dropout episode, F2 status unknown | 1 | 99.9 | **yes** |

## Step 2 -- cumulative dropout fractions (weighted, the frozen basis)

    c1 = (179,390.2 + 58,057.8) / 3,354,119.3
       =        237,447.9       / 3,354,119.3   =  0.070793

    c2 = (227,088.9 + 14,887.4 + 83,502.1 + 99.9) / 3,354,119.3
       =                325,578.3                 / 3,354,119.3   =  0.097068

    monotonicity check:  c1 = 0.070793  <=  c2 = 0.097068   OK

## Step 3 -- survival -> moments

Absorbing failure, per-period hazard `p`, `gamma = 1`, so an individual survives
period `k` with probability `1 - p` independently across periods:

    S1 = E[1 - p]     = 1 - E[p]                 c1 = 1 - S1 = E[p]
    S2 = E[(1 - p)^2] = 1 - 2E[p] + E[p^2]       c2 = 1 - S2 = 2E[p] - E[p^2]

Solving:

    m0 = E[p]   = c1              = 0.070793
    m1 = E[p^2] = 2*c1 - c2       = 2(0.070793) - 0.097068
                                  = 0.141586 - 0.097068
                                  = 0.044518

Moment-cone checks:

    variance   m1 - m0^2 = 0.044518 - 0.005012 = 0.039506  > 0        OK
    Hausdorff  m1 = 0.044518 <= m0 = 0.070793                          OK
               (equivalently E[p(1-p)] = 0.026275 >= 0)

## Step 4 -- Beta parameters by moment inversion

With `s = alpha + beta`, `m0 = alpha/s` and `m1 = alpha(alpha+1)/(s(s+1))`:

    s     = (m0 - m1) / (m1 - m0^2)
          = (0.070793 - 0.044518) / 0.039506
          =        0.026275       / 0.039506   =  0.665094

    alpha = m0 * s = 0.070793 * 0.665094       =  0.047084
    beta  = s - alpha = 0.665094 - 0.047084    =  0.618011

    positivity: alpha > 0, beta > 0                                    OK

## Step 5 -- round trip

Recomputing the moments from the fitted pair:

    alpha/s                    = 0.047084 / 0.665094           = 0.070793  = m0
    alpha(alpha+1)/(s(s+1))    = 0.047084(1.047084)
                                 / (0.665094 * 1.665094)       = 0.044518  = m1

    maximum absolute round-trip error: 0.00e+00                          OK

**Frozen calibration: `Beta(0.047084, 0.618011)`, mean risk 0.070793.**

## Unweighted sensitivity (reported, not the frozen basis)

    c1 = 0.054397   c2 = 0.076528
    m0 = 0.054397   m1 = 0.032267
    alpha = 0.041075   beta = 0.714011   mean risk 0.054397

The unweighted mean risk is 1.6 points lower, as expected given ELS:2002's
oversampling design. The weighted figures are the frozen calibration because
they are the population-representative ones.
