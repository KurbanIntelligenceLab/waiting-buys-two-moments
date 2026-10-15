"""Frozen-config loading and stable configuration IDs.

Every independent run gets an ID derived deterministically from its parameters,
so a sweep is restartable: a run whose ID already has an output row is skipped
rather than recomputed.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import yaml

from .distributions import from_spec
from .welfare import PopulationConfig

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = REPO_ROOT / "experiments" / "config" / "frozen_benchmark.yaml"


def load_frozen(path=None) -> dict:
    with open(path or CONFIG_PATH, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def config_id(**params) -> str:
    """Stable 12-hex-character ID for a parameter dict."""
    payload = json.dumps(params, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


@dataclass(frozen=True)
class PopulationCell:
    """One population-level configuration of the benchmark."""

    dist_name: str
    gamma: int
    T: int
    beta: float
    dist: object

    @property
    def cid(self) -> str:
        return config_id(dist=self.dist_name, gamma=self.gamma,
                         T=self.T, beta=self.beta)

    def population(self) -> PopulationConfig:
        return PopulationConfig(self.dist, self.gamma, self.T, self.beta)

    def as_row(self) -> dict:
        return {"config_id": self.cid, "distribution": self.dist_name,
                "gamma": self.gamma, "T": self.T, "beta": self.beta}


def population_cells(cfg=None) -> list:
    """The full population grid implied by the frozen config.

    Includes both the `resolved` families (recovered from the draft) and the
    `author_frozen` families (chosen for this benchmark on 2026-08-21 and marked
    as such in the config, never presented as recovered values).
    """
    cfg = cfg or load_frozen()
    specs = list(cfg["distributions"]["resolved"]["families"])
    specs += list(cfg["distributions"].get("author_frozen", {}).get("families", []))
    dists = [from_spec(spec) for spec in specs]
    cells = []
    for d in dists:
        for gamma in cfg["observation_model"]["gamma"]:
            for T in cfg["horizon"]["T"]:
                for beta in cfg["budget_ratio"]["beta"]:
                    cells.append(PopulationCell(d.name, int(gamma), int(T),
                                                float(beta), d))
    return cells


def distribution_coverage(cfg=None) -> str:
    cfg = cfg or load_frozen()
    n = len(cfg["distributions"]["resolved"]["families"])
    n += len(cfg["distributions"].get("author_frozen", {}).get("families", []))
    return "full" if n >= 5 else "beta_only"
