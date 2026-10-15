"""Timing rules.

A rule maps the planner's observed prefix to an allocation date:

    rule(data: PrefixData, ctx: RuleContext) -> RuleOutput

`ctx` carries what the planner is assumed to know (T, beta, gamma, prefix t) so
that misspecification experiments can perturb it without touching the data, and
carries a *separate* rule seed so randomised rules never consume the simulation
stream.  Every rule receives the same `PrefixData` object.

Only the two reference rules the brief defines outright are implemented.  The
plug-in rules, the two worst-case rules, and the regret-calibrated randomized
rule are deliberately absent: the last is undefined in the brief (it is the
paper's own contribution) and the worst-case pair is named but not specified.
See NEEDS_DECISION.md items D1 and D8.  The registry raises rather than
substituting a guess, so no benchmark can quietly run against an invented method.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .welfare import PopulationConfig


@dataclass(frozen=True)
class RuleContext:
    """What the rule is told about the world (may be misspecified on purpose)."""

    T: int
    beta: float
    gamma: int
    t: int
    rule_seed: int = 0
    truth: object = None          # PopulationConfig; oracle only


@dataclass
class RuleOutput:
    selected_date: int
    success: bool = True
    failure_reason: str = ""
    n_iterations: int = 0
    randomized: bool = False
    detail: dict = None


def oracle(data, ctx: RuleContext) -> RuleOutput:
    """Reference rule 1: act at t*.  Uses the mean-field truth by construction."""
    if ctx.truth is None:
        return RuleOutput(selected_date=-1, success=False,
                          failure_reason="oracle needs the mean-field truth")
    return RuleOutput(selected_date=int(ctx.truth.t_star))


def act_immediately(data, ctx: RuleContext) -> RuleOutput:
    """Reference rule 2: always allocate at date 1."""
    return RuleOutput(selected_date=1)


class BlockedRule:
    """Placeholder for a rule whose definition is not available."""

    def __init__(self, name: str, reason: str):
        self.name, self.reason = name, reason

    def __call__(self, data, ctx):
        raise NotImplementedError(
            f"timing rule {self.name!r} is not implemented: {self.reason}")


REGISTRY = {
    "oracle": oracle,
    "act_immediately": act_immediately,
    "parametric_beta": BlockedRule(
        "parametric_beta",
        "plug-in decision template not specified; see NEEDS_DECISION D1/D8"),
    "nnls": BlockedRule(
        "nnls",
        "plug-in decision template not specified; see NEEDS_DECISION D1/D8"),
    "npmle": BlockedRule(
        "npmle",
        "plug-in decision template not specified; see NEEDS_DECISION D1/D8"),
    "worst_case_envelope": BlockedRule(
        "worst_case_envelope", "rule undefined; see NEEDS_DECISION D8"),
    "worst_case_lp": BlockedRule(
        "worst_case_lp", "rule undefined; see NEEDS_DECISION D8"),
    "regret_calibrated": BlockedRule(
        "regret_calibrated",
        "the paper's proposed method is undefined in the brief; "
        "see NEEDS_DECISION D1"),
}

AVAILABLE = [k for k, v in REGISTRY.items() if not isinstance(v, BlockedRule)]
BLOCKED = [k for k, v in REGISTRY.items() if isinstance(v, BlockedRule)]


def evaluate(selected_date: int, truth: PopulationConfig) -> dict:
    """Score a selected date against the mean-field truth."""
    s = int(selected_date)
    eff = truth.efficiency(s)
    regret = truth.relative_regret(s)
    return {
        "selected_date": s,
        "true_optimal_date": truth.t_star,
        "exact_argmax_match": bool(s == truth.t_star),
        "true_welfare_at_selected": truth.welfare_at(s),
        "optimal_welfare": truth.W_star,
        "welfare_efficiency": eff,
        "relative_regret": regret,
        "squared_regret": regret ** 2,
    }
