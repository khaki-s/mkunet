"""ROI-restricted legal editing of the four nested BraTS labels over arms P/R/RS0/RS.

One Rule per region h (0 WT, 1 TC, 2 ET) carries its own arm, blend alpha, form
and threshold. A Policy composes at most one rule per region; the three regions
are applied together with the v1 adjacent-legal transitions:

    label L in {0 BG, 1 ED, 2 NCR, 3 ET}; region h contains L > h
    s      = alpha * m_h + (1 - alpha) * q_h,  m_h in {q_h (P), r_h (R), rs0_h, rs_h}
    delete L = h+1 and s < t  -> h          (form delete / recal)
    add    L = h   and s >= t -> h+1        (form add / recal)
    joint  ET deleted and TC score < 0.5 -> ET becomes ED directly (ET rule only)
    a voxel that is both added and deleted keeps its label; nothing outside the ROI moves.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

ARMS = ("P", "R", "RS0", "RS")
FORMS = ("delete", "add", "recal")
THRESHOLDS = (0.30, 0.40, 0.50, 0.60, 0.70)


@dataclass(frozen=True)
class Rule:
    head: int
    form: str
    threshold: float
    arm: str = "P"
    alpha: float = 1.0
    joint: bool = False

    @property
    def delete(self):
        return self.threshold if self.form in ("delete", "recal") else None

    @property
    def add(self):
        return self.threshold if self.form in ("add", "recal") else None

    @property
    def family(self):
        """Selection family: everything except the numeric threshold."""
        return (self.head, self.arm, self.alpha, self.form, self.joint)

    def validate(self):
        if (self.head not in (0, 1, 2) or self.form not in FORMS or self.arm not in ARMS
                or not 0 < self.alpha <= 1 or not 0 < self.threshold < 1
                or (self.arm == "P" and self.alpha != 1.0)
                or (self.joint and (self.head != 2 or self.delete is None))):
            raise ValueError(f"Invalid edit rule {self}")
        return self

    def to_json(self):
        return asdict(self)

    @classmethod
    def from_json(cls, data):
        return cls(**data).validate()


@dataclass(frozen=True)
class Policy:
    rules: tuple = ()

    @classmethod
    def compose(cls, rules):
        rules = tuple(sorted((r for r in rules if r is not None), key=lambda r: r.head))
        if len({r.head for r in rules}) != len(rules):
            raise ValueError("At most one rule per region")
        return cls(rules)

    @property
    def heads(self):
        return tuple(r.head for r in self.rules)

    @property
    def arms(self):
        return {r.arm for r in self.rules}

    def to_json(self):
        return {"rules": [r.to_json() for r in self.rules]}

    @classmethod
    def from_json(cls, data):
        return cls.compose(Rule.from_json(r) for r in data["rules"])


def head_candidates(head, arms=ARMS):
    """Candidate rules for one region; index 0 is None (closed = baseline).

    3 forms x 5 thresholds per (arm, alpha); P uses alpha=1, the others 0.5 and 1.
    With all four arms: 15 x (1+2+2+2) + 1 = 106 per region, plus joint ET variants.
    """
    result = [None]
    for arm in arms:
        for alpha in ((1.0,) if arm == "P" else (0.5, 1.0)):
            for form in FORMS:
                for t in THRESHOLDS:
                    result.append(Rule(head, form, t, arm, alpha))
                    if head == 2 and form != "add":
                        result.append(Rule(head, form, t, arm, alpha, joint=True))
    return result


class Window:
    """Baseline labels, nested prior and arm scores at the ROI voxels of one window.

    Only ROI voxels can change, so every candidate is evaluated on these (N,) vectors;
    ``scatter`` puts the edited labels back into the full window.
    """

    def __init__(self, base, prior, arms):
        self.base, self.prior, self.arms = base, prior, arms

    @classmethod
    def from_window(cls, base, roi, prior, arms):
        roi = np.asarray(roi, bool)
        if prior.shape != (3, *base.shape) or roi.shape != base.shape:
            raise ValueError("Prior or ROI misaligned with the window")
        for name, volume in arms.items():
            if name not in ARMS[1:] or volume.shape != (3, *base.shape):
                raise ValueError(f"Arm {name} misaligned with the window")
        picked = {k: np.asarray(v[:, roi], np.float32) for k, v in arms.items()}
        if not all(np.isfinite(v).all() for v in picked.values()):
            raise ValueError("Non-finite arm scores")
        return cls(np.asarray(base[roi], np.uint8), np.asarray(prior[:, roi], np.float32), picked)

    def score(self, head, arm, alpha):
        q = self.prior[head]
        if arm == "P":
            return q
        if arm not in self.arms:
            raise ValueError(f"Arm {arm} has no scores in this window")
        return alpha * self.arms[arm][head] + (1.0 - alpha) * q

    def apply(self, policy):
        labels = self.base
        result = labels.copy()
        if not policy.rules:
            return result
        up = np.zeros((3, labels.size), bool)
        down = np.zeros((3, labels.size), bool)
        to_edema = np.zeros(labels.size, bool)
        for rule in policy.rules:
            h = rule.head
            s = self.score(h, rule.arm, rule.alpha)
            if rule.delete is not None:
                down[h] = (labels == h + 1) & (s < rule.delete)
            if rule.add is not None:
                up[h] = (labels == h) & (s >= rule.add)
            if rule.joint:
                to_edema = down[2] & (self.score(1, rule.arm, rule.alpha) < 0.5)
        for label in range(4):
            add = up[label] if label < 3 else np.zeros(labels.size, bool)
            delete = down[label - 1] if label else np.zeros(labels.size, bool)
            result[add & ~delete] = label + 1
            result[delete & ~add] = label - 1
        result[to_edema] = 1
        return result

    @staticmethod
    def scatter(base_window, roi, roi_labels):
        edited = np.asarray(base_window, np.uint8).copy()
        edited[np.asarray(roi, bool)] = roi_labels
        return edited
