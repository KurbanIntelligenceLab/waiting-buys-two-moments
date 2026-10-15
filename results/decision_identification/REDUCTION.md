# Decision identification: the reduction and why the certificates are valid

This document states and proves every step that the decision-identification
experiment (`wtm/decision_id.py`, `experiments/scripts/decision_identification.py`)
relies on. It was written before the computation was run. Each lemma is also
checked numerically in `experiments/tests/test_decision_id.py`, which is noted
under each one, but the proofs below are what the conclusions rest on.

Scope: gamma = 1, the paper's mean-field model, and the paper's population-level
information set. Nothing here changes the simulator, the estimators, the existing
benchmarks or the manuscript.

## 0. Setting and notation

- A configuration is (P, T, beta): a risk distribution P on [0,1], a horizon T,
  and a budget ratio beta in (0,1). The candidate dates are
  S = {1, ..., T-1}, the same set `PopulationConfig.dates` uses. Date T is
  excluded because u^T = 0.
- The prefix length is tau, and n = 2 tau - 1. The information set is
  M = { Q probability measure on [0,1] : E_Q[p^j] = E_P[p^j], j = 0..n }.
  Equivalently, E_Q[p^j (1-p)^(n-j)] = E_P[p^j (1-p)^(n-j)] for j = 0..n. The
  two constraint sets are the same because {p^j (1-p)^(n-j)} is a basis of the
  polynomials of degree <= n. The code uses the second form because it is
  better conditioned.
- For a date s and group k = 0..s:
  phi_{s,k}(p) = C(s,k) p^k (1-p)^(2s-1-k) and u^s(p) = 1 - (1-p)^(T-s).
  m_{s,k}(Q) = E_Q[phi_{s,k}] and v_{s,k}(Q) = E_Q[phi_{s,k} u^s].
- Welfare is W^s(Q) = max { sum_k a_k v_{s,k}(Q) : sum_k a_k m_{s,k}(Q) <= beta,
  a in [0,1]^(s+1) }, which `wtm/welfare.py` computes by greedy fill.
- Facts used throughout: 0 <= u^s <= 1, and u^s is nondecreasing in p. Every
  phi_{s,k} >= 0 on [0,1]. Therefore 0 <= v_{s,k} <= m_{s,k}.

**Topology.** M is convex and weak-* compact. It is a closed subset of the
probability measures on the compact set [0,1], cut out by finitely many
continuous linear constraints. Each m_{s,k} and v_{s,k} is weak-* continuous in
Q. W^s(Q) is the minimum over lambda in [0,1] of functions that are jointly
continuous in (lambda, Q) (Lemma 2). Such a minimum is continuous in Q, so
every supremum over M below is attained.

## 1. The three decision targets

For d in S:

| Target | Definition |
|---|---|
| guaranteed optimal | for all Q in M and all r in S: W^d(Q) >= W^r(Q) |
| uniquely optimal | for all Q in M and all r != d: W^d(Q) > W^r(Q) |
| identified earliest optimal | for all Q in M: d = min argmax_s W^s(Q) |

The third target is the paper's tie-breaking rule. Ties go to the earliest date,
which is also what `numpy.argmax` does in `PopulationConfig.t_star`.

Define the **largest advantage of r over d**:

    V(d, r) = max_{Q in M} [ W^r(Q) - W^d(Q) ]      (attained, by compactness)

**Proposition 1.**
- (a) d is guaranteed optimal iff V(d, r) <= 0 for every r.
- (b) d is uniquely optimal iff V(d, r) < 0 for every r != d.
- (c) d is the identified earliest optimal date iff V(d, r) < 0 for every
  r < d and V(d, r) <= 0 for every r > d.
- (d) unique implies earliest-identified, which implies guaranteed.
- (e) Any date with property (a), (b) or (c) lies in D*(P) = argmax_s W^s(P).
  A date with property (c) equals min D*(P).

*Proof.*
- (a) Both quantifiers are universal and commute. For each r, "for all Q,
  W^r(Q) - W^d(Q) <= 0" is exactly V(d, r) <= 0.
- (b) If V(d, r) < 0, then W^r(Q) - W^d(Q) <= V(d, r) < 0 for every Q.
  Conversely, the maximum is attained at some Q*, and W^r(Q*) < W^d(Q*) gives
  V(d, r) < 0.
- (c) d = min argmax W(Q) means W^d(Q) > W^r(Q) for r < d and
  W^d(Q) >= W^r(Q) for r > d. Apply the arguments of (a) and (b) to each r.
- (d) Immediate from (a) to (c).
- (e) P is in M. A guaranteed optimal d satisfies W^d(P) >= W^r(P) for all r.
  Under (c), d must be the earliest maximizer for Q = P. ∎

**Quantifier note.** All three properties are universal statements over Q, so
checking them one competitor at a time is exact. The opposite property, "some
single Q makes r optimal", is existential over Q. It needs one Q that beats every
competitor at once, and pairwise checks are only necessary conditions for it.
The experiment never uses such a property.

**Consequence for the computation.** It suffices to take each d in D*(P), which
is computed exactly in rational arithmetic (Section 7), and bound V(d, r) for
every r != d.

## 2. The welfare representation

**Lemma 1 (threshold representation).** For every Q, define the nested
allocation a(theta) for theta in [0, s+1] by

    a_k(theta) = clip(theta - (s - k), 0, 1)        k = 0..s.

Here theta = 0 treats nobody, theta in [0,1] partly treats group s, and
theta = s+1 treats everyone. Then

    W^s(Q) = max { a(theta) · v_s(Q) : theta in [0, s+1],  a(theta) · m_s(Q) <= beta }.

*Proof.* The right side maximizes over a subset of the feasible allocations, so
it is at most W^s(Q). For the reverse inequality, three steps.

*Step 1: groups are ordered by posterior benefit.* For groups k < k' with
positive mass, let rho_k = v_{s,k}/m_{s,k}. Then

    v_{k'} m_k - v_k m_{k'}
      = (1/2) ∫∫ (u(p) - u(q)) [phi_{k'}(p) phi_k(q) - phi_k(p) phi_{k'}(q)] dQ(p) dQ(q).

With e = k' - k, the bracket equals

    C(s,k') C(s,k) (pq)^k ((1-p)(1-q))^(2s-1-k') [ (p(1-q))^e - (q(1-p))^e ].

The exponent 2s-1-k' is >= 0 because k' <= s. Since p(1-q) - q(1-p) = p - q,
the last factor has the sign of p - q. So does u(p) - u(q), since u is
nondecreasing. The integrand is therefore >= 0, and rho_{k'} >= rho_k. This
holds for every pair of groups with positive mass, not only adjacent ones.

*Step 2: the greedy fill is optimal.* Let a* fill groups from k = s downward
until the budget binds, with a possible fractional group k*. Take the dual price
lambda* = rho_{k*} if the budget binds with m_{k*} > 0, and lambda* = 0
otherwise. Set mu_k = (v_k - lambda* m_k)_+. By Step 1,
v_k - lambda* m_k >= 0 for k > k* and <= 0 for k < k*.
- If the budget binds: sum_k a*_k v_k = lambda* beta + sum_k mu_k. This is the
  dual objective of Lemma 2, so a* is optimal by weak duality.
- If the budget does not bind: a* treats everyone and lambda* = 0.

*Step 3: the greedy fill has threshold form.* Groups with m_{s,k} = 0 have
v_{s,k} = 0, since 0 <= v <= m, so their coordinate of a* can be set to 1
without changing the value or the budget used. After that change, a* = a(theta*)
with theta* = (number of fully treated groups) + (fill of group k*). ∎

*Numerical check:* `test_threshold_matches_greedy` compares the two sides on
random discrete Q, including Q with empty groups and beta above the total
active mass.

**Lemma 2 (dual form, bounded price).** For every Q,

    W^s(Q) = min_{lambda in [0,1]} f_s(lambda, Q),
    f_s(lambda, Q) = lambda beta + sum_k (v_{s,k}(Q) - lambda m_{s,k}(Q))_+ .

*Proof.* This is LP duality for the knapsack LP:
min lambda beta + sum mu_k subject to mu_k >= v_k - lambda m_k, mu >= 0,
lambda >= 0. The LP is feasible and bounded, so strong duality holds and the
dual minimum is attained over lambda >= 0. The price can be restricted to
[0,1]: since v_k <= m_k, f_s(1, Q) = beta, and for lambda >= 1,
f_s(lambda, Q) >= lambda beta >= beta. ∎

*Numerical check:* `test_dual_form`.

**Proposition 3 (joint maximization).** For d != r,

    V(d, r) = sup over (Q, theta, lambda) in M × [0, r+1] × [0,1]
              with a(theta) · m_r(Q) <= beta
              of   G(Q, theta, lambda) = a(theta) · v_r(Q) - lambda beta
                                          - sum_k (v_{d,k}(Q) - lambda m_{d,k}(Q))_+ .

*Proof.* Fix Q. By Lemma 1, the supremum over feasible theta of
a(theta) · v_r(Q) is W^r(Q). By Lemma 2, the supremum over lambda of
-f_d(lambda, Q) is -W^d(Q). The two terms involve separate variables, so the
joint supremum over (theta, lambda) is W^r(Q) - W^d(Q). Taking the supremum
over Q gives V(d, r). ∎

## 3. Bounding a box of (theta, lambda)

**Lemma 4 (box relaxation).** Let B = [theta_l, theta_u] × [lambda_l, lambda_u]
be a sub-box of the domain. Define

    A_B  = sum_k a_k(theta_u) phi_{r,k} u^r        (objective kernel)
    C_B  = sum_k a_k(theta_l) phi_{r,k}            (relaxed budget kernel)
    D_k  = phi_{d,k} (u^d - lambda_u),  k = 0..d

and let

    U*(B) = sup_{Q in M, E_Q[C_B] <= beta}  E_Q[A_B] - lambda_l beta
                                            - sum_k ( E_Q[D_k] )_+ .

Then G(Q, theta, lambda) <= U*(B) for every (theta, lambda) in B and every
Q in M with a(theta) · m_r(Q) <= beta.

*Proof.* a(theta) is coordinatewise nondecreasing in theta, and phi, u >= 0.
Therefore:
- a(theta) · v_r <= E[A_B].
- a(theta_l) · m_r <= a(theta) · m_r <= beta, so Q is feasible for the relaxed
  budget constraint.
- -lambda beta <= -lambda_l beta.
- m_{d,k} >= 0 means v_{d,k} - lambda m_{d,k} is nonincreasing in lambda, so
  -(v_{d,k} - lambda m_{d,k})_+ <= -(v_{d,k} - lambda_u m_{d,k})_+
  = -(E_Q[D_k])_+. ∎

**Corollary.** If boxes B_1, ..., B_L cover [0, r+1] × [0,1], then
V(d, r) <= max_i U*(B_i). A box whose relaxed budget constraint is infeasible
on M has U*(B) = -infinity.

## 4. A certificate valid on all of [0,1]

U*(B) is a linear program over measures on the whole interval [0,1]:

    maximize  ∫A_B dQ + sum_k z_k - lambda_l beta
    subject to  ∫ p^j (1-p)^(n-j) dQ = mu_j   (j = 0..n),   ∫C_B dQ <= beta,
                z_k <= 0,   z_k + ∫D_k dQ <= 0,   Q >= 0 on [0,1].

The substitution is exact: for fixed Q, the best z_k is min(0, -E[D_k]).

**Lemma 5 (dual certificate).** Let y in R^(n+1), eta >= 0 and
omega in [0,1]^(d+1) be arbitrary. Define the polynomial

    R(p) = sum_j y_j p^j (1-p)^(n-j) + eta C_B(p) + sum_k omega_k D_k(p) - A_B(p).

If L is any number with R(p) >= L for all p in [0,1], then

    U*(B) <= U_cert(B) := sum_j y_j mu_j + eta beta - lambda_l beta + max(0, -L).

*Proof.* Let (Q, z) be feasible. Since z_k <= 0 and 1 - omega_k >= 0,
z_k <= omega_k z_k. Since omega_k >= 0 and z_k <= -E[D_k],
omega_k z_k <= -omega_k E[D_k]. Therefore

    E[A_B] + sum z_k <= E[A_B - sum omega_k D_k]
                      = E[ sum_j y_j p^j(1-p)^(n-j) + eta C_B - R ]
                      <= sum_j y_j mu_j + eta E[C_B] - L          (R >= L, Q(1) = 1)
                      <= sum_j y_j mu_j + eta beta + max(0, -L).

The last step uses eta >= 0 and E[C_B] <= beta. Subtract lambda_l beta. ∎

The inequality R >= L is required on **all of [0,1]**, not on a grid. Q ranges
over every probability measure on [0,1], so no grid enters the validity
argument. Grids are used only to *find* good multipliers (y, eta, omega), by
solving the discretized LP and taking its dual. The multipliers are then
converted to exact rationals and checked exactly. A poor or wrong-signed dual
from the solver can only produce a weak or failing certificate. It can never
produce an invalid one.

**Lemma 6 (Bernstein lower bound).** Write a polynomial R of degree <= N on
[a,b] in the Bernstein basis of that interval: R = sum_i beta_i B_{i,N}. Then
min_{[a,b]} R >= min_i beta_i.

*Proof.* The B_{i,N} are nonnegative on [a,b] and sum to 1. ∎

The checker converts R exactly to Bernstein form on [0,1]. It then subdivides
the interval with exact integer de Casteljau steps, repeatedly splitting the
subinterval with the smallest coefficient. It stops when that lower bound is
good enough or a depth limit is reached. L is the minimum coefficient over the
final partition, so it is a rigorous lower bound on R over [0,1].

**Theorem 7 (certified strict inequality).** Suppose:
- the leaves of a bisection tree cover [0, r+1] × [0,1] exactly. The checker
  verifies this exactly: every leaf lies in the domain and the leaf areas sum
  to r+1.
- each leaf B has an exact rational certificate with U_cert(B) < 0.

Then V(d, r) < 0.

*Proof.* Combine Lemma 4 and Lemma 5, then Proposition 3. ∎

Every quantity in the check is an exact rational: mu_j (the P moments), beta,
the box endpoints, the multipliers and the polynomial coefficients. The
integer-coefficient polynomials come from `observation.MixedPoly`. No
floating-point number enters the conclusion.

## 5. Witnesses: proving that a date is not optimal

**Lemma 8 (exact witness).** Let Q be a probability measure on [0,1] with
E_Q[p^j] = E_P[p^j] exactly for j <= n, and suppose W^r(Q) > W^d(Q), evaluated
exactly. Then V(d, r) > 0. If instead W^r(Q) = W^d(Q) exactly, then
V(d, r) >= 0.

The experiment builds candidate witnesses from the LP. It takes an LP solution
A, an atomic measure with rational atoms and weights whose moments are off by a
small error e. It then forms

    Q = (1 - eps) A + eps B,    dB = (1 + g) dP,

where g is a polynomial of degree <= n. The coefficients of g solve the exact
Hankel system E_P[p^j g] = -(1 - eps) e_j / eps, j = 0..n, which makes the
moments of Q exact. Q is a valid probability measure when 1 + g >= 0 on
[0,1]. That condition is checked exactly with the same Bernstein routine. Its
group masses and utilities are exact rationals. They combine A's atoms with the
exact moments of P, which is possible because
E_B[K] = E_P[K] + sum_i c_i E_P[K p^i]. W^r(Q) and W^d(Q) then come from an
exact rational greedy fill (Lemma 1). The experiment tries several eps values
and keeps any Q that passes every exact check. A witness counts only if every
check passes.

The true population P is itself a member of M. When D*(P) has several dates,
P is an exact tie witness: for d, d' in D*(P), V(d, d') >= 0.

## 6. Branch and bound, and what "undecided" means

For each d in D*(P) and each r != d, the procedure is as follows.

1. Start with the boxes [j, j+1] × [0,1], j = 0..r. Compute a floating-point
   upper bound for each box. The dual comes from a grid LP, and the grid is
   refined by exchange: points where R is most negative on a fine test grid
   are added and the LP is re-solved.
2. Keep lower bounds from point evaluations. At a single (theta, lambda),
   Lemma 4 is exact, and the grid LP's optimizer is a candidate witness.
3. Repeatedly split the box with the largest upper bound. Split theta if the
   theta-side gap dominates and lambda otherwise.
4. Stop in one of three ways:
   - **Certified:** every leaf has U < 0 in floating point, and each leaf then
     passes the exact check (Theorem 7). Leaves that fail the exact check are
     split again.
   - **Witness:** a point lower bound is positive and an exact witness is built
     (Lemma 8).
   - **Undecided:** the box budget or the minimum box width is exhausted.

Floating-point values guide the search only. The labels "certified" and
"witness" require the exact checks. Everything else is reported as undecided,
together with the best exact lower bound and the best float upper bound found.
In particular, a pair where V(d, r) is exactly zero, or numerically
indistinguishable from zero, can never be certified from above. It stays
undecided unless an exact tie witness exists.

## 7. Classification of a configuration–prefix pair

D*(P) is computed exactly: W^s(P) for every s, from P's exact rational moments
and an exact greedy fill.

| Outcome | Condition | Implies |
|---|---|---|
| **CERTIFIED_UNIQUE** | some d in D*(P) has certified V(d, r) < 0 for every r != d | d is uniquely optimal, hence also earliest-identified and guaranteed (Prop. 1(d)) |
| **NO_GUARANTEED_DATE** | every d in D*(P) has an exact witness Q with W^r(Q) > W^d(Q) for some r | no date is guaranteed, earliest-identified or unique (Prop. 1(e)) |
| **TIE_AT_TRUTH** | abs(D*(P)) >= 2 (exact) | uniqueness is refuted by P itself. Guaranteed and earliest are decided only by witnesses. An exact zero cannot be certified from above, so absent a strict witness they are reported undecided |
| **UNDECIDED** | none of the above | reported with the pairs (d, r) left open and their bounds |

The three targets can differ only through exact ties. Certification here always
proves strict inequalities, so a guaranteed-but-not-unique date is never
*certified* by this method. If one exists, it appears as TIE_AT_TRUTH or
UNDECIDED. The report states this rather than folding the cases together.
