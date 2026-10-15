"""Decision identification: certify, refute, or leave open "date d is optimal
for every population in M_tau".

The mathematics is in `results/decision_identification/REDUCTION.md`, and the
section numbers below refer to it.  Division of labour:

  * floating point (HiGHS on a refinable grid) *searches*: it proposes dual
    multipliers for each (theta, lambda) box and candidate witness populations;
  * exact rational arithmetic *decides*: a box counts as certified only when its
    multipliers, converted to rationals, pass the Bernstein nonnegativity check
    on all of [0, 1] (Lemmas 5-6), and a witness counts only when its moments
    match exactly and its welfare difference is exactly positive (Lemma 8).

Nothing here modifies or depends on the simulator or the estimators.
"""

from __future__ import annotations

import heapq
import itertools
import math
import time
from dataclasses import dataclass, field
from fractions import Fraction
from math import comb

import numpy as np
from scipy.optimize import linprog

from .observation import MixedPoly, phi, utility_poly

GAMMA = 1


# ---------------------------------------------------------------------------
# exact polynomial helpers
# ---------------------------------------------------------------------------
def mono(poly: MixedPoly) -> list:
    """Exact integer monomial coefficients."""
    return [int(c) for c in poly.to_monomial()]


def mixed_basis(n: int, j: int) -> MixedPoly:
    """p^j (1-p)^(n-j)."""
    return MixedPoly.from_dict({(j, n - j): 1})


def eval_mixed_exact(poly: MixedPoly, x: Fraction) -> Fraction:
    one_minus = 1 - x
    return sum((c * x ** i * one_minus ** j for (i, j), c in poly.coeffs), Fraction(0))


def poly_add(acc: list, coeffs, scale) -> None:
    if len(acc) < len(coeffs):
        acc.extend([Fraction(0)] * (len(coeffs) - len(acc)))
    if scale == 0:
        return
    for i, c in enumerate(coeffs):
        if c:
            acc[i] += scale * c


def mono_to_bernstein(c: list) -> list:
    """Bernstein coefficients on [0,1] of sum_j c_j p^j (degree N = len(c)-1).

    b_i = sum_{j <= i} C(i, j) / C(N, j) * c_j.
    """
    N = len(c) - 1
    out = []
    for i in range(N + 1):
        s = Fraction(0)
        for j in range(i + 1):
            if c[j]:
                s += Fraction(comb(i, j), comb(N, j)) * c[j]
        out.append(s)
    return out


def _split_int(b: list, N: int):
    """Exact de Casteljau split at the midpoint of integer coefficients.

    Input coefficients have (implicit) common denominator D; outputs have
    common denominator D * 2^N.
    """
    work = list(b)
    left = [work[0] << N]
    right_rev = [work[-1] << N]
    for level in range(1, N + 1):
        work = [work[i] + work[i + 1] for i in range(len(work) - 1)]
        left.append(work[0] << (N - level))
        right_rev.append(work[-1] << (N - level))
    return left, right_rev[::-1]


def bernstein_lower_bound(bern: list, target: Fraction | None = None,
                          max_leaves: int = 4000, max_depth: int = 60) -> Fraction:
    """Rigorous lower bound on min_{[0,1]} of the polynomial with these
    Bernstein coefficients (Lemma 6), refined by exact subdivision.

    Stops as soon as the bound exceeds `target` (if given), when the minimum
    coefficient of the worst leaf is an endpoint value (then it is exact), or
    when the leaf / depth budget is exhausted.  The value returned is always a
    valid lower bound.
    """
    N = len(bern) - 1
    den = 1
    for x in bern:
        den = den * x.denominator // math.gcd(den, x.denominator)
    ints = [int(x * den) for x in bern]
    # heap of (float key, id, ints, log2 extra denominator, depth)
    counter = itertools.count()
    heap = [(min(ints) / den, next(counter), ints, 0, 0)]
    n_leaves = 1
    while True:
        key, _, b, e, depth = heap[0]
        lo = Fraction(min(b), den << e)
        if target is not None and lo >= target:
            return lo
        m = min(b)
        if m == b[0] or m == b[-1] or depth >= max_depth or n_leaves >= max_leaves:
            # exact minimum at an endpoint, or budget exhausted: the heap minimum
            # is the bound (heap key is a float; recompute exactly over leaves)
            return min(Fraction(min(bb), den << ee) for _, _, bb, ee, _ in heap)
        heapq.heappop(heap)
        L, R = _split_int(b, N)
        for part in (L, R):
            heapq.heappush(heap, (min(part) / (den << (e + N)), next(counter),
                                  part, e + N, depth + 1))
        n_leaves += 1


def exact_greedy(masses: list, utils: list, beta: Fraction) -> Fraction:
    """Exact greedy top-down fill (Lemma 1)."""
    remaining = beta
    total = Fraction(0)
    for k in range(len(masses) - 1, -1, -1):
        if remaining <= 0:
            break
        m = masses[k]
        if m <= 0:
            continue
        take = min(remaining, m)
        total += utils[k] * take / m
        remaining -= take
    return total


def nested_alloc(theta, s: int):
    """a_k(theta) = clip(theta - (s - k), 0, 1), k = 0..s (Lemma 1)."""
    return [min(max(theta - (s - k), 0), 1) for k in range(s + 1)]


def solve_exact(A: list, b: list) -> list:
    """Exact Gaussian elimination for a small square rational system."""
    n = len(b)
    M = [list(row) + [bb] for row, bb in zip(A, b)]
    for col in range(n):
        piv = next(r for r in range(col, n) if M[r][col] != 0)
        M[col], M[piv] = M[piv], M[col]
        pv = M[col][col]
        M[col] = [x / pv for x in M[col]]
        for r in range(n):
            if r != col and M[r][col] != 0:
                f = M[r][col]
                M[r] = [x - f * y for x, y in zip(M[r], M[col])]
    return [M[r][n] for r in range(n)]


# ---------------------------------------------------------------------------
# the problem
# ---------------------------------------------------------------------------
def base_grid() -> np.ndarray:
    geo = np.geomspace(1e-8, 2e-2, 80)
    g = np.concatenate([np.linspace(0.0, 1.0, 1201), geo, 1.0 - geo])
    return np.unique(np.clip(g, 0.0, 1.0))


def test_grid() -> np.ndarray:
    geo = np.geomspace(1e-12, 5e-2, 600)
    g = np.concatenate([np.linspace(0.0, 1.0, 40001), geo, 1.0 - geo])
    return np.unique(np.clip(g, 0.0, 1.0))


class DecisionProblem:
    """One (distribution, T, beta, tau) configuration-prefix pair."""

    def __init__(self, dist, T: int, beta: float, tau: int, first_date: int = 1):
        self.dist, self.T, self.tau = dist, int(T), int(tau)
        self.beta = float(beta)
        self.beta_q = Fraction(str(beta))
        self.n = 2 * self.tau - 1
        # candidate dates: the paper's {1, ..., T-1} by default; first_date = tau
        # restricts to dates still available after a prefix of length tau
        self.first_date = max(1, int(first_date))
        self.dates = list(range(self.first_date, self.T))
        self.basis = [mixed_basis(self.n, j) for j in range(self.n + 1)]
        self.mu_q = [dist.mixed_exact(j, self.n - j) for j in range(self.n + 1)]
        self.mu = np.array([float(x) for x in self.mu_q])
        self._phi, self._uphi, self._mono = {}, {}, {}
        self._basis_mono = [mono(b) for b in self.basis]
        self._exact_E = {}

    # -- kernels -------------------------------------------------------------
    def phi_polys(self, s):
        if s not in self._phi:
            u = utility_poly(s, self.T)
            self._phi[s] = [phi(s, k, GAMMA) for k in range(s + 1)]
            self._uphi[s] = [p * u for p in self._phi[s]]
        return self._phi[s], self._uphi[s]

    def mono_kernels(self, s):
        if s not in self._mono:
            P, U = self.phi_polys(s)
            self._mono[s] = ([mono(p) for p in P], [mono(q) for q in U])
        return self._mono[s]

    def eval_kernels(self, s, x):
        P, U = self.phi_polys(s)
        return (np.vstack([p.evaluate(x) for p in P]),
                np.vstack([q.evaluate(x) for q in U]))

    def eval_basis(self, x):
        return np.vstack([b.evaluate(x) for b in self.basis])

    # -- exact truth -----------------------------------------------------------
    def exact_expect(self, poly: MixedPoly) -> Fraction:
        key = poly.coeffs
        if key not in self._exact_E:
            self._exact_E[key] = poly.expectation_exact(self.dist)
        return self._exact_E[key]

    def truth_welfare_exact(self) -> dict:
        out = {}
        for s in self.dates:
            P, U = self.phi_polys(s)
            m = [self.exact_expect(p) for p in P]
            v = [self.exact_expect(q) for q in U]
            out[s] = exact_greedy(m, v, self.beta_q)
        return out

    def truth_argmax(self):
        W = self.truth_welfare_exact()
        best = max(W.values())
        return [s for s in self.dates if W[s] == best], W


# ---------------------------------------------------------------------------
# box LP (floating point, search only)
# ---------------------------------------------------------------------------
@dataclass
class GridState:
    """Grid of candidate support points for one (d, r) job, refined by exchange."""

    prob: DecisionProblem
    d: int
    r: int
    x: np.ndarray = None
    xt: np.ndarray = None

    def __post_init__(self):
        self.x = base_grid()
        self._fill()
        self.xt = test_grid()
        self.Bt = self.prob.eval_basis(self.xt)
        self.PHId_t, self.UPHId_t = self.prob.eval_kernels(self.d, self.xt)
        self.PHIr_t, self.UPHIr_t = self.prob.eval_kernels(self.r, self.xt)

    def _fill(self):
        self.B = self.prob.eval_basis(self.x)
        self.PHId, self.UPHId = self.prob.eval_kernels(self.d, self.x)
        self.PHIr, self.UPHIr = self.prob.eval_kernels(self.r, self.x)

    def add_points(self, pts):
        pts = np.setdiff1d(np.asarray(pts, float), self.x)
        if len(pts):
            self.x = np.sort(np.concatenate([self.x, pts]))
            self._fill()
        return len(pts)


LP_ATTEMPTS = (("highs", {}), ("highs-ds", {"presolve": False}),
               ("highs-ipm", {"presolve": False}))


def _box_floats(box):
    return tuple(float(v) for v in box)


def solve_box_lp(gs: GridState, box, penalty: float = 1e3):
    """Grid LP for U*(B) in units of beta; returns primal, dual and value."""
    prob, d, r = gs.prob, gs.d, gs.r
    tl, tu, ll, lu = _box_floats(box)
    beta = prob.beta
    G = len(gs.x)
    aU = np.array(nested_alloc(tu, r), float)
    aL = np.array(nested_alloc(tl, r), float)
    A = aU @ gs.UPHIr / beta
    C = aL @ gs.PHIr / beta
    D = (gs.UPHId - lu * gs.PHId) / beta                      # (d+1, G)
    K = d + 1
    # variables: w (G), z (K), sigma (1); minimise the negative objective
    c = np.concatenate([-A, -np.ones(K), [penalty]])
    A_eq = np.hstack([gs.B / prob.mu[:, None], np.zeros((prob.n + 1, K + 1))])
    b_eq = np.ones(prob.n + 1)
    A_ub = np.vstack([
        np.concatenate([C, np.zeros(K), [-1.0]])[None, :],
        np.hstack([D, np.eye(K), np.zeros((K, 1))]),
    ])
    b_ub = np.concatenate([[1.0], np.zeros(K)])
    bounds = [(0, None)] * G + [(None, 0)] * K + [(0, None)]
    # HiGHS presolve occasionally fails ("Not Set") on refined grids; the same LP
    # then solves without presolve.  Retries affect only which multipliers are
    # proposed, never the validity of a certificate (Lemma 5).
    for method, options in LP_ATTEMPTS:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds,
                      method=method, options=options)
        if res.status == 0:
            break
    else:
        return None
    ys = -res.eqlin.marginals            # multipliers of scaled moment rows
    eta = max(0.0, -res.ineqlin.marginals[0])
    omega = np.clip(-res.ineqlin.marginals[1:], 0.0, 1.0)
    # unscaled multipliers (Lemma 5 units): y_j = beta * ys_j / mu_j
    y = beta * ys / prob.mu
    value = -res.fun - ll                # in units of beta, excluding -lambda_l
    dual_obj = (ys.sum() + eta) - ll     # in units of beta
    return {"w": res.x[:G], "x": gs.x,
            "K": (gs.PHIr, gs.UPHIr, gs.PHId, gs.UPHId),
            "sigma": res.x[-1], "y": y, "eta": eta,
            "omega": omega, "value": value, "dual_obj": dual_obj}


def residual_on_test(gs: GridState, box, y, eta, omega):
    """R(p)/beta on the fine test grid (float, guidance only)."""
    tl, tu, ll, lu = _box_floats(box)
    r, beta = gs.r, gs.prob.beta
    aU = np.array(nested_alloc(tu, r), float)
    aL = np.array(nested_alloc(tl, r), float)
    R = (y @ gs.Bt + eta * (aL @ gs.PHIr_t) + omega @ (gs.UPHId_t - lu * gs.PHId_t)
         - aU @ gs.UPHIr_t)
    return R / beta


def float_box_bound(gs: GridState, box, max_exchange: int = 6, tol: float = 1e-11):
    """Float upper bound on U*(B)/beta with exchange refinement of the grid."""
    sol = None
    for _ in range(max_exchange + 1):
        sol = solve_box_lp(gs, box)
        if sol is None:
            return None
        R = residual_on_test(gs, box, sol["y"], sol["eta"], sol["omega"])
        neg = R < -tol
        if not neg.any():
            break
        # local minima among negative points
        idx = np.where(neg)[0]
        Rl = np.r_[np.inf, R, np.inf]
        loc = idx[(R[idx] <= Rl[idx]) & (R[idx] <= Rl[idx + 2])]
        loc = loc[np.argsort(R[loc])][:30]
        if gs.add_points(gs.xt[loc]) == 0:
            break
    Rmin = float(R.min())
    sol["float_U"] = sol["dual_obj"] + max(0.0, -Rmin)
    sol["float_Rmin"] = Rmin
    return sol


# ---------------------------------------------------------------------------
# exact certificate check (Lemmas 5-6, Theorem 7)
# ---------------------------------------------------------------------------
def verify_box_certificate(prob: DecisionProblem, d: int, r: int, box, y, eta, omega,
                           max_leaves: int = 4000):
    """Exact check.  Returns (U_cert as Fraction, lower bound L, parts)."""
    tl, tu, ll, lu = box                                  # Fractions
    yq = [Fraction(float(v)) for v in y]
    etaq = Fraction(float(max(0.0, eta)))
    omq = [min(max(Fraction(float(v)), Fraction(0)), Fraction(1)) for v in omega]
    PHIr, UPHIr = prob.mono_kernels(r)
    PHId, UPHId = prob.mono_kernels(d)
    Rc = []
    for j, bm in enumerate(prob._basis_mono):
        poly_add(Rc, bm, yq[j])
    for k, a in enumerate(nested_alloc(tl, r)):
        poly_add(Rc, PHIr[k], etaq * a)
    for k, a in enumerate(nested_alloc(tu, r)):
        poly_add(Rc, UPHIr[k], -a)
    for k in range(d + 1):
        poly_add(Rc, UPHId[k], omq[k])
        poly_add(Rc, PHId[k], -omq[k] * lu)
    base = sum((yj * mj for yj, mj in zip(yq, prob.mu_q)), Fraction(0)) \
        + etaq * prob.beta_q - ll * prob.beta_q
    bern = mono_to_bernstein(Rc)
    # we need L > base (base must be negative for a useful certificate)
    target = base / 2 if base < 0 else Fraction(0)
    L = bernstein_lower_bound(bern, target=target, max_leaves=max_leaves)
    U = base + max(Fraction(0), -L)
    return U, L, {"y": yq, "eta": etaq, "omega": omq, "base": base}


# ---------------------------------------------------------------------------
# exact witnesses (Lemma 8)
# ---------------------------------------------------------------------------
def welfare_exact_mixture(prob: DecisionProblem, s: int, atoms, weights, eps, g):
    """Exact W^s of Q = (1-eps) * sum w_i delta_{x_i} + eps * (1+g) dP."""
    P, U = prob.phi_polys(s)
    m, v = [], []
    for pk, uk in zip(P, U):
        mA = sum((w * eval_mixed_exact(pk, x) for x, w in zip(atoms, weights)), Fraction(0))
        vA = sum((w * eval_mixed_exact(uk, x) for x, w in zip(atoms, weights)), Fraction(0))
        mB = prob.exact_expect(pk) + sum(
            (c * prob.exact_expect(pk * MixedPoly.from_dict({(i, 0): 1}))
             for i, c in enumerate(g) if c), Fraction(0))
        vB = prob.exact_expect(uk) + sum(
            (c * prob.exact_expect(uk * MixedPoly.from_dict({(i, 0): 1}))
             for i, c in enumerate(g) if c), Fraction(0))
        m.append((1 - eps) * mA + eps * mB)
        v.append((1 - eps) * vA + eps * vB)
    return exact_greedy(m, v, prob.beta_q)


def build_exact_witness(prob: DecisionProblem, d: int, r: int, x, w,
                        eps_list=(1e-9, 1e-7, 1e-5, 1e-3, 1e-2, 5e-2)):
    """Try to turn a float grid measure (x, w) into an exact member of M with
    W^r > W^d (strictly).  Returns a dict or None."""
    keep = w > 1e-15
    if not keep.any():
        return None
    atoms = [Fraction(float(v)) for v in x[keep]]
    wts = [Fraction(float(v)) for v in w[keep]]
    n = prob.n
    momA = [sum((wt * eval_mixed_exact(prob.basis[j], a) for a, wt in zip(atoms, wts)),
                Fraction(0)) for j in range(n + 1)]
    e = [momA[j] - prob.mu_q[j] for j in range(n + 1)]
    H = [[prob.dist.mixed_exact(j + i, n - j) for i in range(n + 1)] for j in range(n + 1)]
    for eps_f in eps_list:
        eps = Fraction(eps_f)
        rhs = [-(1 - eps) * ej / eps for ej in e]
        g = solve_exact(H, rhs)
        # 1 + g >= 0 on [0, 1], exactly
        gpoly = [Fraction(1) + g[0]] + list(g[1:])
        if bernstein_lower_bound(mono_to_bernstein(gpoly), target=Fraction(0)) < 0:
            continue
        # exact moment check (redundant by construction, verified anyway)
        for j in range(n + 1):
            mB = prob.mu_q[j] + sum((c * prob.dist.mixed_exact(j + i, n - j)
                                     for i, c in enumerate(g)), Fraction(0))
            assert (1 - eps) * momA[j] + eps * mB == prob.mu_q[j]
        Wr = welfare_exact_mixture(prob, r, atoms, wts, eps, g)
        Wd = welfare_exact_mixture(prob, d, atoms, wts, eps, g)
        if Wr > Wd:
            return {"atoms": atoms, "weights": wts, "eps": eps, "g": g,
                    "W_r": Wr, "W_d": Wd, "advantage": Wr - Wd}
    return None


def float_welfare(prob, s, x, w):
    PHI, UPHI = prob.eval_kernels(s, x)
    m, v = PHI @ w, UPHI @ w
    rem, tot = prob.beta, 0.0
    for k in range(len(m) - 1, -1, -1):
        if rem <= 0:
            break
        if m[k] <= 0:
            continue
        take = min(rem, m[k])
        tot += v[k] * take / m[k]
        rem -= take
    return tot


# ---------------------------------------------------------------------------
# branch and bound for one (d, r) pair (Section 6)
# ---------------------------------------------------------------------------
@dataclass
class PairResult:
    d: int
    r: int
    status: str                      # certified | witness | undecided
    n_boxes: int = 0
    n_leaves: int = 0
    max_float_upper: float = float("nan")   # over open boxes (undecided) in units of beta
    worst_certified_U: float = float("nan") # max exact U_cert over leaves, units of beta
    truth_gap: Fraction = None              # W^r(P) - W^d(P), exact
    witness: dict = None
    leaves: list = field(default_factory=list)
    seconds: float = 0.0
    note: str = ""


def _split(box, which):
    tl, tu, ll, lu = box
    if which == "theta":
        m = (tl + tu) / 2
        return (tl, m, ll, lu), (m, tu, ll, lu)
    m = (ll + lu) / 2
    return (tl, tu, ll, m), (tl, tu, m, lu)


def _split_choice(prob, gs, box, sol):
    """Split the dimension whose relaxation gap is larger at the LP solution."""
    tl, tu, ll, lu = _box_floats(box)
    w = sol["w"]
    PHIr, UPHIr, PHId, UPHId = sol["K"]
    r = gs.r
    aU = np.array(nested_alloc(tu, r), float)
    aL = np.array(nested_alloc(tl, r), float)
    gap_t = float(((aU - aL) @ UPHIr) @ w) + 0.5 * float(((aU - aL) @ PHIr) @ w)
    md = PHId @ w
    vd = UPHId @ w
    gap_l = (lu - ll) * prob.beta + float(np.sum(np.maximum(vd - ll * md, 0)
                                                 - np.maximum(vd - lu * md, 0)))
    if tu - tl < 2.0 ** -30:
        return "lambda"
    if lu - ll < 2.0 ** -34:
        return "theta"
    return "theta" if gap_t >= gap_l else "lambda"


def bound_pair(prob: DecisionProblem, d: int, r: int, max_boxes: int = 3000,
               time_limit: float = 900.0, witness_tol: float = 1e-9,
               search_witness: bool = True) -> PairResult:
    """Certify V(d, r) < 0, find an exact witness V(d, r) > 0, or give up."""
    t0 = time.time()
    gs = GridState(prob, d, r)
    res = PairResult(d=d, r=r, status="undecided")
    counter = itertools.count()
    heap = []                     # (-float_U, id, box, sol, exact_failed)
    certified = []
    n_eval = 0
    beta = prob.beta

    def try_witness(sol):
        if sol is None or not search_witness:
            return None
        x, w = sol["x"], sol["w"]
        diff = float_welfare(prob, r, x, w) - float_welfare(prob, d, x, w)
        if diff / beta > witness_tol:
            return build_exact_witness(prob, d, r, x, w)
        return None

    def evaluate(box):
        nonlocal n_eval
        n_eval += 1
        sol = float_box_bound(gs, box)
        if sol is None:
            return None
        return sol

    def process(box, parent_U=np.inf):
        """Evaluate a box; certify it, find a witness, or queue it."""
        sol = evaluate(box)
        if sol is None:
            # no bound available: keep the parent's bound as its priority
            heapq.heappush(heap, (-parent_U, next(counter), box, None, True))
            return None
        if sol["float_U"] < 0:
            U, L, parts = verify_box_certificate(prob, d, r, box, sol["y"], sol["eta"],
                                                 sol["omega"])
            if U < 0:
                certified.append((box, U, parts))
                return None
            heapq.heappush(heap, (-1e-30, next(counter), box, sol, True))
            return None
        wit = try_witness(sol)
        if wit is not None:
            return wit
        heapq.heappush(heap, (-sol["float_U"], next(counter), box, sol, False))
        return None

    for j in range(r + 1):
        wit = process((Fraction(j), Fraction(j + 1), Fraction(0), Fraction(1)))
        if wit is not None:
            res.status, res.witness = "witness", wit
            break

    while res.status == "undecided" and heap:
        if n_eval >= max_boxes or time.time() - t0 > time_limit:
            res.note = "budget exhausted"
            break
        negU, _, box, sol, failed = heapq.heappop(heap)
        tl, tu, ll, lu = box
        if (tu - tl) < Fraction(1, 2 ** 32) and (lu - ll) < Fraction(1, 2 ** 36):
            heapq.heappush(heap, (negU, next(counter), box, sol, failed))
            res.note = "minimum box size reached"
            break
        # point evaluation at the box centre: exact for the relaxation at a point
        if sol is not None and not failed:
            c = ((tl + tu) / 2, (tl + tu) / 2, (ll + lu) / 2, (ll + lu) / 2)
            psol = solve_box_lp(gs, c)
            wit = try_witness(psol)
            if wit is not None:
                res.status, res.witness = "witness", wit
                break
        which = _split_choice(prob, gs, box, sol) if sol is not None else "theta"
        for child in _split(box, which):
            wit = process(child, -negU)
            if wit is not None:
                res.status, res.witness = "witness", wit
                break

    if res.status == "undecided" and not heap:
        area = sum(((b[1] - b[0]) * (b[3] - b[2]) for b, _, _ in certified), Fraction(0))
        inside = all(b[0] >= 0 and b[1] <= r + 1 and b[2] >= 0 and b[3] <= 1
                     for b, _, _ in certified)
        if area == r + 1 and inside:
            res.status = "certified"
        else:
            res.note = f"coverage check failed: area={area}, inside={inside}"
    res.n_boxes = n_eval
    res.n_leaves = len(certified)
    if certified:
        res.worst_certified_U = float(max(U for _, U, _ in certified)) / beta
    if res.status == "undecided" and heap:
        res.max_float_upper = float(-heap[0][0])
    res.leaves = certified
    res.seconds = time.time() - t0
    return res
