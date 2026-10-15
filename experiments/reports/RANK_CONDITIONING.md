# Experiment 1: the two ranks, and why they diverge at high degree

Both ranks are retained in `results/structural_rank.csv` and neither supersedes
the other:

| Column | Meaning |
|---|---|
| `exact_rank` | Gaussian elimination over GF(p) for two large primes (2^31-1, 2^61-1). Exact arithmetic. |
| `exact_rank_primes_agree` | whether both primes returned the same rank |
| `matrix_rank` | floating-point SVD rank at the default tolerance |
| `rank_at_tol_x{0.1,1,10,100}` | SVD rank at four tolerance multipliers |
| `rank_stable_across_tolerances` | whether all four agree |
| `rank_agrees_with_exact` | whether the SVD rank equals the exact rank |

## The headline result (gamma = 1)

For `t = 1..10`, both regimes, both ranks agree exactly and are stable across
all four tolerance multipliers:

- cohort counts: rank `2t` -- Lemma 1 confirmed;
- individual exit records: rank `2t+1` -- Remark 1's extra dimension confirmed.

No conditioning caveat applies here. The monomial degree stays at most 21 and the
matrices are well enough scaled after row normalisation.

## Where the two ranks diverge, and what that is

At `gamma = 4` and `t >= 6`, and at `gamma = 2` for `t = 10` in the
individual-record regime, the floating-point SVD rank falls **below** the exact
rank (at `gamma = 4`, `t = 10`: SVD 34 vs exact 44), and
`rank_stable_across_tolerances` turns false.

**This is monomial-basis conditioning, not a structural property of the
observable span, and it must not be reported as one.** The mechanism:

- the functionals are represented for this computation in the monomial basis, by
  expanding `p^i (1-p)^j` -- the only basis in which "rank of the span of
  polynomials" is a matrix rank;
- that expansion is alternating, and its coefficients grow with degree. At
  `gamma = 4`, `t = 10` the monomial degree reaches `gamma*t + t - 1 = 49`, and
  the coefficients span far more than the ~16 decimal digits a double carries;
- the resulting matrix has singular values decaying past machine precision, so an
  SVD cannot resolve the smaller ones from rounding noise at any tolerance. The
  span itself is unaffected.

The exact GF(p) rank is immune to this: it is integer arithmetic on the same
exact coefficients, and the two independent primes agree.

**Therefore:** at `gamma != 1` and high `t`, read `exact_rank`. The SVD column is
retained as a diagnostic of the conditioning, and the tolerance sweep is retained
so the instability is visible rather than hidden behind one arbitrary cut.

Note that this is a statement about *our chosen representation for the rank
computation*, not about the identification problem. The welfare and LP machinery
never touches the monomial basis: it works in the mixed basis `p^i (1-p)^j`,
where every `phi_{s,k}` has nonnegative coefficients at every gamma and
expectations are cancellation-free sums of positive terms.

## Measured growth rate at integer gamma

Cohort-count `exact_rank` by prefix length:

| gamma | ranks, t = 1..10 | asymptotic growth |
|---|---|---|
| 1 | 2, 4, 6, 8, 10, 12, 14, 16, 18, 20 | **2 per period** (= `2t` exactly) |
| 2 | 2, 5, 8, 11, 14, 17, 20, 23, 26, 29 | **3 per period** (= `3t-1` exactly) |
| 4 | 2, 5, 9, 14, 19, 24, 29, 34, 39, 44 | **5 per period** from `t = 5` on |

The pattern is `gamma+1` dimensions per period. At `gamma = 4` the rank equals
the *number of functionals* (`t(t+3)/2`) for `t <= 4` -- the family is linearly
independent there and the count, not the span, is binding -- and grows by 5 per
period from `t = 5`.

This is offered as a measurement, not as a theorem. Section 9 of the draft
regards the general-gamma case as open, and nothing here closes it: the
computation covers integer `gamma` only, where the observable functionals remain
polynomials. For non-integer `gamma` the span involves genuine fractional powers
of `1-p` and a matrix rank in the monomial basis is not the right object at all.
