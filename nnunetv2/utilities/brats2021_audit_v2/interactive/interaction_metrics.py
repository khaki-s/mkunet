"""Patient-level metrics for separate frozen interactive-segmentation experiments.

Internal Dataset137 labels are 0=BG, 1=ED, 2=NCR, 3=ET.  NCR is the
necrotic/non-enhancing core combined label.  Undefined both-empty Dice values
remain ``None``; cohort coverage and paired denominators are always explicit.
This module does not load images, models, ground-truth files or checkpoints.
"""
from __future__ import annotations

from collections import Counter
import math
from numbers import Integral, Real

import numpy as np


REGIONS = ("WT", "TC", "ET", "ED", "NCR")
TARGETS = ("ED", "ET", "NCR", "joint")
TIME_KEYS = (
    "cumulative_interaction_seconds",
    "cumulative_selection_seconds",
    "cumulative_inference_seconds",
)


def prompt_budget_metadata(kind):
    """Make annotation costs explicit; paired rounds are not single clicks."""
    if kind not in ('point', 'box', 'point_box'):
        raise ValueError('prompt_kind must be point, box, or point_box')
    paired = kind == 'point_box'
    return {
        'budget_unit': 'rounds' if paired else 'annotations',
        'annotations_per_round': 2 if paired else 1,
        'points_per_round': int(kind in ('point', 'point_box')),
        'boxes_per_round': int(kind in ('box', 'point_box')),
        'round_definition': ('one same-error point and 2D box, one head/polarity, one model prediction'
                             if paired else 'one new annotation, one model prediction'),
        'actual_tool_calls_definition': 'upstream prediction calls; autozoom/refinement may run multiple network forwards',
        'history_replay_counts_as_new_annotations': False,
    }


def _annotation_counts(state, kind):
    """Legacy point/box records predate explicit point/box/round fields."""
    rounds = state.get('actual_rounds', state['actual_prompts'])
    return {
        'actual_rounds': rounds,
        'actual_points': state.get('actual_points', rounds if kind == 'point' else 0),
        'actual_boxes': state.get('actual_boxes', rounds if kind == 'box' else 0),
    }


def _labels(values, name):
    array = np.asarray(values)
    if array.ndim != 3 or not np.issubdtype(array.dtype, np.integer):
        raise ValueError(f"{name} must be a 3D integer label volume")
    if not np.isin(array, (0, 1, 2, 3)).all():
        raise ValueError(f"{name} must use internal labels 0=BG,1=ED,2=NCR,3=ET")
    return array


def region_metrics(pred, gt, voxel_volume_mm3=1.0):
    """Return count-based Dice/FP/FN for all five regions, with no empty coercion."""
    prediction, truth = _labels(pred, "prediction"), _labels(gt, "GT")
    if prediction.shape != truth.shape:
        raise ValueError("Prediction and GT shapes must match")
    if (isinstance(voxel_volume_mm3, bool) or not isinstance(voxel_volume_mm3, Real)
            or not math.isfinite(float(voxel_volume_mm3)) or voxel_volume_mm3 <= 0):
        raise ValueError("voxel_volume_mm3 must be finite and positive")
    volume = float(voxel_volume_mm3)
    definitions = {
        "WT": lambda a: a > 0,
        "TC": lambda a: a >= 2,
        "ET": lambda a: a == 3,
        "ED": lambda a: a == 1,
        "NCR": lambda a: a == 2,
    }
    output = {}
    for name, select in definitions.items():
        p, g = select(prediction), select(truth)
        tp = int(np.count_nonzero(p & g))
        fp = int(np.count_nonzero(p & ~g))
        fn = int(np.count_nonzero(~p & g))
        denominator = 2 * tp + fp + fn
        output[name] = {
            "dice": 2.0 * tp / denominator if denominator else None,
            "tp": tp, "fp": fp, "fn": fn, "n_ref": tp + fn, "n_pred": tp + fp,
            "fp_mm3": fp * volume, "fn_mm3": fn * volume,
        }
    return output


def _mean(values):
    return float(np.mean(values)) if values else None


def _ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def _summary(values):
    if not values:
        return {"mean": None, "median": None, "p95": None}
    return {"mean": float(np.mean(values)), "median": float(np.median(values)),
            "p95": float(np.percentile(values, 95))}


def _unique_strings(values, name):
    values = list(values)
    if not values or any(not isinstance(v, str) or not v for v in values):
        raise ValueError(f"{name} must contain nonempty strings")
    if len(values) != len(set(values)):
        raise ValueError(f"{name} contains duplicates")
    return values


def _validate_metrics(metrics, context):
    if not isinstance(metrics, dict) or any(r not in metrics for r in REGIONS):
        raise ValueError(f"{context}: all five region metrics are required")
    for region in REGIONS:
        row = metrics[region]
        if not isinstance(row, dict):
            raise ValueError(f"{context}/{region}: expected a metrics object")
        for key in ("tp", "fp", "fn", "n_ref", "n_pred"):
            value = row.get(key)
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
                raise ValueError(f"{context}/{region}: invalid count {key}")
        if row["n_ref"] != row["tp"] + row["fn"] or row["n_pred"] != row["tp"] + row["fp"]:
            raise ValueError(f"{context}/{region}: inconsistent region counts")
        denominator = row["n_ref"] + row["n_pred"]
        dice = row.get("dice")
        if denominator == 0:
            if dice is not None:
                raise ValueError(f"{context}/{region}: both-empty Dice must be None")
        elif (isinstance(dice, bool) or not isinstance(dice, Real)
              or not math.isfinite(float(dice))
              or not math.isclose(float(dice), 2 * row["tp"] / denominator, abs_tol=1e-9)):
            raise ValueError(f"{context}/{region}: Dice must match region counts")
        for key in ("fp_mm3", "fn_mm3"):
            value = row.get(key)
            if (isinstance(value, bool) or not isinstance(value, Real)
                    or not math.isfinite(float(value)) or value < 0):
                raise ValueError(f"{context}/{region}: invalid physical volume {key}")


def aggregate_results(records, expected_case_ids, targets, budgets=(0, 1, 3, 5), tolerance=1e-6):
    """Aggregate one prompt modality without hiding failed or absent patients.

    ``records`` contains one object per (case_id, target), either ``complete`` or
    ``failed``.  Complete objects have ``baseline`` (five region metrics) and
    ``budgets`` keyed by str(budget).  Each budget contains ``metrics``, integer
    ``actual_prompts``/``actual_tool_calls``, all three cumulative TIME_KEYS and
    boolean ``carried_forward``.  Failed records may include a valid baseline.

    Returned ``groups`` and ``timing_groups`` are flat CSV-compatible rows.
    ``mean_delta_pp`` uses only patients with BOTH finite baseline/current Dice;
    it must not be reconstructed by subtracting independently averaged scores.
    ``tolerance`` is an absolute Dice difference, not a percentage-point value.
    Mixed point/box/point_box records are rejected, even when case IDs differ.
    Paired budgets count rounds, require explicit counts, and allow two new
    annotations per round. Point and box retain their original budget meaning.
    """
    expected = _unique_strings(expected_case_ids, "expected_case_ids")
    target_list = _unique_strings(targets, "targets")
    if any(target not in TARGETS for target in target_list):
        raise ValueError(f"targets must be drawn from {TARGETS}")
    budget_list = list(budgets)
    if (not budget_list or any(isinstance(b, bool) or not isinstance(b, Integral) or b < 0 for b in budget_list)
            or budget_list != sorted(set(budget_list)) or budget_list[0] != 0):
        raise ValueError("budgets must be sorted unique nonnegative integers beginning with zero")
    if (isinstance(tolerance, bool) or not isinstance(tolerance, Real)
            or not math.isfinite(float(tolerance)) or tolerance < 0):
        raise ValueError("tolerance must be finite and nonnegative")
    expected_set, targets_set = set(expected), set(target_list)
    lookup, kinds = {}, set()
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("Every result must be a dictionary")
        case, target = record.get("case_id"), record.get("target")
        if case not in expected_set or target not in targets_set:
            raise ValueError(f"Unexpected case/target in results: {case!r}/{target!r}")
        key = (case, target)
        if key in lookup:
            raise ValueError(f"Duplicate case/target result: {case}/{target}")
        status = record.get("status")
        if status not in ("complete", "failed"):
            raise ValueError("Result status must be 'complete' or 'failed'")
        kind = record.get("prompt_kind")
        design = prompt_budget_metadata(kind)
        for field in ('budget_unit', 'annotations_per_round', 'points_per_round', 'boxes_per_round'):
            if field in record and record[field] != design[field]:
                raise ValueError(f'{field} does not match prompt_kind')
        if status == 'complete' and kind == 'point_box' and record.get('budget_unit') != 'rounds':
            raise ValueError('point_box records must explicitly declare budget_unit=rounds')
        kinds.add(kind)
        if "baseline" in record and record["baseline"] is not None:
            _validate_metrics(record["baseline"], f"{case}/{target}/baseline")
        if status == "complete":
            if not record.get("baseline"):
                raise ValueError(f"{case}/{target}: complete result requires baseline")
            states = record.get("budgets")
            if not isinstance(states, dict):
                raise ValueError(f"{case}/{target}: complete result requires budget states")
            prior_prompts, prior_calls = 0, 0
            prior_times = {key: 0.0 for key in TIME_KEYS}
            for budget in budget_list:
                state = states.get(str(budget))
                if not isinstance(state, dict):
                    raise ValueError(f"{case}/{target}: missing requested budget {budget}")
                _validate_metrics(state.get("metrics"), f"{case}/{target}/{budget}")
                for region in REGIONS:
                    if state["metrics"][region]["n_ref"] != record["baseline"][region]["n_ref"]:
                        raise ValueError(f"{case}/{target}: GT region size changed between budgets")
                for field in ("actual_prompts", "actual_tool_calls"):
                    value = state.get(field)
                    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
                        raise ValueError(f"{case}/{target}/{budget}: invalid {field}")
                if state["actual_prompts"] > budget * design['annotations_per_round']:
                    raise ValueError(f"{case}/{target}: actual prompts exceed budget")
                explicit_fields = ('actual_rounds', 'actual_points', 'actual_boxes')
                if kind == 'point_box' or any(field in state for field in explicit_fields):
                    for field in explicit_fields:
                        value = state.get(field)
                        if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
                            raise ValueError(f'{case}/{target}/{budget}: invalid {field}')
                    rounds = state['actual_rounds']
                    if (rounds > budget or state['actual_tool_calls'] != rounds
                            or state['actual_points'] != rounds * design['points_per_round']
                            or state['actual_boxes'] != rounds * design['boxes_per_round']
                            or state['actual_prompts'] != state['actual_points'] + state['actual_boxes']):
                        raise ValueError(f'{case}/{target}/{budget}: inconsistent annotation/round counts')
                if state.get('budget_unit', design['budget_unit']) != design['budget_unit']:
                    raise ValueError(f'{case}/{target}/{budget}: inconsistent budget_unit')
                if state["actual_prompts"] < prior_prompts or state["actual_tool_calls"] < prior_calls:
                    raise ValueError(f"{case}/{target}: cumulative prompt/call counts decreased")
                if not isinstance(state.get("carried_forward"), bool):
                    raise ValueError(f"{case}/{target}: carried_forward must be boolean")
                for field in TIME_KEYS:
                    value = state.get(field)
                    if (isinstance(value, bool) or not isinstance(value, Real)
                            or not math.isfinite(float(value)) or value < 0
                            or value < prior_times[field] - 1e-9):
                        raise ValueError(f"{case}/{target}/{budget}: invalid/decreasing {field}")
                    prior_times[field] = float(value)
                if budget == 0:
                    if state["actual_prompts"] or state["actual_tool_calls"]:
                        raise ValueError("Budget zero must contain no prompts or tool calls")
                    for region in REGIONS:
                        for field in ("tp", "fp", "fn"):
                            if state["metrics"][region][field] != record["baseline"][region][field]:
                                raise ValueError("Budget-zero metrics must match baseline")
                prior_prompts, prior_calls = state["actual_prompts"], state["actual_tool_calls"]
        lookup[key] = record
    if len(kinds) > 1:
        raise ValueError("Point, box and point_box results must be aggregated in separate experiments")

    # Ground-truth volumes are patient properties, regardless of target condition.
    truth_counts = {}
    for (case, target), record in lookup.items():
        if record.get("baseline"):
            counts = tuple(record["baseline"][r]["n_ref"] for r in REGIONS)
            if case in truth_counts and truth_counts[case] != counts:
                raise ValueError(f"GT region size differs between target conditions for {case}")
            truth_counts[case] = counts

    kind = next(iter(kinds)) if kinds else None
    design = prompt_budget_metadata(kind) if kind is not None else {}
    groups, timing_rows, by_target = [], [], {}
    for target in target_list:
        complete = [lookup[(case, target)] for case in expected
                    if (case, target) in lookup and lookup[(case, target)]["status"] == "complete"]
        failed = [case for case in expected
                  if (case, target) in lookup and lookup[(case, target)]["status"] == "failed"]
        missing = [case for case in expected if (case, target) not in lookup]
        available_baselines = [lookup[(case, target)] for case in expected
                               if (case, target) in lookup and lookup[(case, target)].get("baseline")]
        by_target[target] = {
            "expected": len(expected), "complete": len(complete), "failed": len(failed),
            "missing": len(missing), "failed_case_ids": failed, "missing_case_ids": missing,
            "complete_case_ids": [r["case_id"] for r in complete],
            "baseline_available": len(available_baselines),
            "full_cohort_complete": not failed and not missing,
        }
        for budget in budget_list:
            states = [record["budgets"][str(budget)] for record in complete]
            timings = {
                "target": target, "prompt_kind": kind, "budget": int(budget),
                "budget_unit": design.get('budget_unit'),
                "n_expected": len(expected), "n_complete": len(complete),
                "n_failed": len(failed), "n_missing": len(missing),
                "full_cohort_complete": not failed and not missing,
                "carried_forward_n": sum(s["carried_forward"] for s in states),
            }
            for field in TIME_KEYS + ("actual_prompts", "actual_tool_calls"):
                for statistic, value in _summary([float(s[field]) for s in states]).items():
                    timings[f"{field}_{statistic}"] = value
            for field in ('actual_rounds', 'actual_points', 'actual_boxes'):
                for statistic, value in _summary([float(_annotation_counts(s, kind)[field])
                                                  for s in states]).items():
                    timings[f'{field}_{statistic}'] = value
            retreat = {r: [] for r in ("WT", "TC")}
            for record, state in zip(complete, states):
                for region in retreat:
                    before, after = record["baseline"][region]["dice"], state["metrics"][region]["dice"]
                    if before is not None and after is not None:
                        retreat[region].append(after - before < -tolerance)
            for region, values in retreat.items():
                timings[f"{region}_retreat_paired_n"] = len(values)
                timings[f"{region}_retreat_n"] = int(sum(values))
                timings[f"{region}_retreat_fraction"] = _ratio(sum(values), len(values))
            both_known = []
            for record, state in zip(complete, states):
                pairs = [(record["baseline"][r]["dice"], state["metrics"][r]["dice"]) for r in ("WT", "TC")]
                if all(before is not None and after is not None for before, after in pairs):
                    both_known.append(any(after - before < -tolerance for before, after in pairs))
            timings["WT_or_TC_retreat_paired_n"] = len(both_known)
            timings["WT_or_TC_retreat_n"] = int(sum(both_known))
            timings["WT_or_TC_retreat_fraction"] = _ratio(sum(both_known), len(both_known))
            timing_rows.append(timings)

            for region in REGIONS:
                previous = [record["baseline"][region] for record in complete]
                current = [state["metrics"][region] for state in states]
                baseline_all = [record["baseline"][region] for record in available_baselines]
                paired = [(before["dice"], after["dice"]) for before, after in zip(previous, current)
                          if before["dice"] is not None and after["dice"] is not None]
                deltas = [after - before for before, after in paired]
                improved = sum(delta > tolerance for delta in deltas)
                degraded = sum(delta < -tolerance for delta in deltas)
                tied = len(deltas) - improved - degraded
                positive_pairs = [(before, after) for before, after in zip(previous, current) if after["n_ref"] > 0]
                empty_pairs = [(before, after) for before, after in zip(previous, current) if after["n_ref"] == 0]
                finite_after = [row["dice"] for row in current if row["dice"] is not None]
                finite_before = [row["dice"] for row in baseline_all if row["dice"] is not None]
                row = {
                    "target": target, "prompt_kind": kind, "budget": int(budget), "region": region,
                    "budget_unit": design.get('budget_unit'),
                    "n_expected": len(expected), "n_complete": len(complete),
                    "n_failed": len(failed), "n_missing": len(missing),
                    "full_cohort_complete": not failed and not missing,
                    "dice_mean": _mean(finite_after), "dice_valid_n": len(finite_after),
                    "baseline_mean": _mean(finite_before), "baseline_valid_n": len(finite_before),
                    "paired_n": len(paired), "paired_baseline_mean": _mean([p[0] for p in paired]),
                    "paired_dice_mean": _mean([p[1] for p in paired]),
                    "mean_delta_pp": 100 * _mean(deltas) if deltas else None,
                    "improved_n": int(improved), "degraded_n": int(degraded), "unchanged_n": int(tied),
                    "improved_fraction": _ratio(improved, len(deltas)),
                    "degraded_fraction": _ratio(degraded, len(deltas)),
                    "unchanged_fraction": _ratio(tied, len(deltas)),
                    "gt_positive_n": len(positive_pairs),
                    "gt_positive_baseline_available_n": sum(p["n_ref"] > 0 for p in baseline_all),
                    "gt_positive_dice_mean": _mean([p[1]["dice"] for p in positive_pairs]),
                    "gt_positive_baseline_mean": _mean([p[0]["dice"] for p in positive_pairs]),
                    "gt_positive_mean_delta_pp": (100 * _mean([p[1]["dice"] - p[0]["dice"] for p in positive_pairs])
                                                   if positive_pairs else None),
                    "gt_empty_n": len(empty_pairs),
                    "gt_empty_baseline_fp_mean_mm3": _mean([p[0]["fp_mm3"] for p in empty_pairs]),
                    "gt_empty_fp_mean_mm3": _mean([p[1]["fp_mm3"] for p in empty_pairs]),
                    "gt_empty_baseline_fp_positive_n": sum(p[0]["fp"] > 0 for p in empty_pairs),
                    "gt_empty_fp_positive_n": sum(p[1]["fp"] > 0 for p in empty_pairs),
                    "gt_empty_baseline_fp_prevalence": _ratio(sum(p[0]["fp"] > 0 for p in empty_pairs), len(empty_pairs)),
                    "gt_empty_fp_prevalence": _ratio(sum(p[1]["fp"] > 0 for p in empty_pairs), len(empty_pairs)),
                    "gt_empty_fp_improved_n": sum(p[1]["fp"] < p[0]["fp"] for p in empty_pairs),
                    "gt_empty_fp_degraded_n": sum(p[1]["fp"] > p[0]["fp"] for p in empty_pairs),
                    "fp_mean_mm3": _mean([p["fp_mm3"] for p in current]),
                    "fn_mean_mm3": _mean([p["fn_mm3"] for p in current]),
                    "baseline_fp_mean_mm3": _mean([p["fp_mm3"] for p in previous]),
                    "baseline_fn_mean_mm3": _mean([p["fn_mm3"] for p in previous]),
                    "both_empty_n": sum(p["n_ref"] == 0 and p["n_pred"] == 0 for p in current),
                    "dice_undefined_to_defined_n": sum(p["dice"] is None and q["dice"] is not None
                                                       for p, q in zip(previous, current)),
                    "dice_defined_to_undefined_n": sum(p["dice"] is not None and q["dice"] is None
                                                       for p, q in zip(previous, current)),
                }
                groups.append(row)
    status_counts = Counter(record["status"] for record in lookup.values())
    return {
        "schema_version": 1,
        "metric_definition": "patient-equal region Dice; both-empty=None; deltas and change fractions use finite paired Dice",
        "tolerance_dice": float(tolerance),
        "prompt_kind": kind,
        **design,
        "budgets": [int(b) for b in budget_list],
        "coverage": {
            "expected_cases": len(expected), "expected_case_ids": expected,
            "targets": target_list, "expected_case_target_pairs": len(expected) * len(target_list),
            "complete_pairs": status_counts["complete"], "failed_pairs": status_counts["failed"],
            "missing_pairs": len(expected) * len(target_list) - len(lookup),
            "full_cohort_complete": status_counts["complete"] == len(expected) * len(target_list),
            "by_target": by_target,
        },
        "groups": groups,
        "timing_groups": timing_rows,
    }
