"""Apply only approved edits to the four legal, nested BraTS labels."""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np

from .dataset import PROB_FLOOR


@dataclass(frozen=True)
class Rule:
    head: int
    delete: float | None
    add: float | None
    alpha: float = 1.


@dataclass(frozen=True)
class Policy:
    rules: tuple[Rule, ...] = ()
    joint: bool = False

    @property
    def family(self):
        return tuple((r.head, r.delete is not None, r.add is not None) for r in self.rules), self.joint

    @property
    def heads(self):
        return tuple(r.head for r in self.rules)

    def to_json(self):
        return {"joint": self.joint, "rules": [
            {"head": r.head, "delete": r.delete, "add": r.add, "alpha": r.alpha}
            for r in self.rules]}

    @classmethod
    def from_json(cls, data):
        rules = tuple(Rule(**value) for value in data["rules"])
        if len({r.head for r in rules}) != len(rules) or any(
            r.head not in (0, 1, 2) or not 0 < r.alpha <= 1
            or (r.delete is None and r.add is None)
            or (r.delete is not None and not 0 < r.delete <= .5)
            or (r.add is not None and not .5 < r.add <= 1)
            for r in rules
        ):
            raise ValueError("Invalid frozen edit rules")
        joint = data["joint"]
        if type(joint) is not bool or (joint and not {1, 2}.issubset(r.head for r in rules if r.delete is not None)):
            raise ValueError("Joint ET/TC editing needs both deletion rules")
        return cls(rules, joint)


def candidates(heads):
    result = [Policy()]
    for head in heads:
        for alpha in (.5, 1.):
            result.extend(Policy((Rule(head, t, None, alpha),)) for t in (.20, .35, .50))
            result.extend(Policy((Rule(head, None, t, alpha),)) for t in (.65, .80))
            result.extend(Policy((Rule(head, .35, t, alpha),)) for t in (.65, .80))
    if {1, 2}.issubset(heads):
        for tc, et in product((.25, .40, .50), repeat=2):
            result.append(Policy((Rule(1, tc, None), Rule(2, et, None)), joint=True))
    return result


@dataclass
class Window:
    """Only the axial planes queried by the refiner, with scores on native XY."""

    z: np.ndarray
    labels: np.ndarray
    model_probability: np.ndarray
    baseline_probability: np.ndarray

    @classmethod
    def build(cls, labels, records, scores, probability):
        z = np.array(sorted({r["z"] for r in records}), dtype=int)
        dims = (3, *labels.shape[:2], len(z))
        model = np.full(dims, np.nan, dtype=np.float16)
        prior = np.zeros(dims, dtype=np.float16)
        if len(records) != len(scores) or scores.shape != probability.shape or scores.shape[1:] != labels.shape[:2]:
            raise ValueError("Misaligned slice scores and native labels")
        if not np.isfinite(scores).all() or not np.isfinite(probability).all():
            raise ValueError("Expected finite SAM and baseline probabilities")
        if len(records) and (min(z) < 0 or max(z) >= labels.shape[-1]):
            raise ValueError("Score planes are outside the native volume")
        for i, record in enumerate(records):
            head = record["head"]
            plane = int(np.searchsorted(z, record["z"]))
            if head not in (0, 1, 2) or np.isfinite(model[head, :, :, plane]).any():
                raise ValueError("Invalid or duplicate region/slice")
            model[head, :, :, plane] = scores[i]
            prior[head, :, :, plane] = probability[i]
        return cls(z, labels[:, :, z], model, prior)

    def apply(self, policy):
        labels = self.labels
        result = labels.copy()
        if not policy.rules:
            return result
        up = np.zeros((3, *labels.shape), dtype=bool)
        down = np.zeros_like(up)
        for rule in policy.rules:
            h = rule.head
            base = labels > h
            prior = self.baseline_probability[h].astype(np.float32)
            # Keep the blend anchored to the hard label when stored probabilities
            # and the baseline disagree at the decision boundary.
            anchor = np.where(base, np.maximum(prior, .5001), np.minimum(prior, .4999))
            score = rule.alpha * self.model_probability[h].astype(np.float32) + (1 - rule.alpha) * anchor
            if rule.delete is not None:
                down[h] = base & (score < rule.delete)
            if rule.add is not None:
                up[h] = ~base & (score >= rule.add) & (prior >= np.float16(PROB_FLOOR))
        for label in range(4):
            original = labels == label
            add = original & up[label] if label < 3 else np.zeros_like(original)
            delete = original & down[label - 1] if label else np.zeros_like(original)
            if label < 3:
                result[add & ~delete] = label + 1
            if label:
                result[delete & ~add] = label - 1
        if policy.joint:
            result[(labels == 3) & down[2] & down[1]] = 1
        return result

    def apply_volume(self, baseline, policy):
        if not policy.rules:
            return baseline.copy()
        final = baseline.copy()
        final[:, :, self.z] = self.apply(policy)
        for head in set(range(3)) - set(policy.heads):
            if np.any((baseline > head) != (final > head)):
                raise AssertionError("An inactive region was edited")
        return final
