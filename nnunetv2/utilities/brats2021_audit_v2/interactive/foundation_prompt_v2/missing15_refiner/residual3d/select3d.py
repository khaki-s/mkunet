"""Nested val80 selection of per-(combination, region) edit policies, then freeze.

Three allowed arm-sets are each run through the same nested selection so their
out-of-fold gains are paired: {P}, {P,R}, {P,R,RS0,RS}. The frozen method uses
the full arm-set unless it does not beat {P,R} out of fold, in which case {P,R}
is frozen so SAM cannot add only variance.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import csv
from pathlib import Path

import numpy as np
from sklearn.model_selection import KFold

from ...common import check_file, file_identity, read_json, write_json
from ..dataset import HEADS, catalog, heads_for, load_mask, split_manifest
from . import data3d as D
from .policy3d import ARMS, Policy, Window, candidates

SCHEMA = "brats_missing15_residual3d_frozen_v1"
ARM_SETS = {"P": ("P",), "PR": ("P", "R"), "PRS": ("P", "R", "RS0", "RS")}


def policy_source():
    root = Path(__file__).parent
    return {n: file_identity(root / n)["sha256"] for n in ("policy3d.py", "select3d.py", "infer3d.py")}


def dice_counts(tp, denominator):
    return np.divide(200.0 * tp, denominator, out=np.full(3, 100.0), where=denominator != 0)


def counts(prediction, truth):
    tp, denom = [], []
    for h in range(3):
        pred, ref = prediction > h, truth > h
        tp.append(int(np.count_nonzero(pred & ref)))
        denom.append(int(np.count_nonzero(pred)) + int(np.count_nonzero(ref)))
    return np.asarray(tp), np.asarray(denom)


def crop(volume, box):
    return volume[tuple(slice(a, b) for a, b in box)]


def load_arms(entry, sam_entry):
    with np.load(entry["scores"]["path"], allow_pickle=False) as f:
        region, prior_region = f["region"], f["prior_region"]
    arms = {"R": region.astype(np.float32)}
    if sam_entry is not None:
        with np.load(sam_entry["scores"]["path"], allow_pickle=False) as f:
            arms["RS"] = f["rs"].astype(np.float32)
            arms["RS0"] = f["rs0"].astype(np.float32)
    return prior_region.astype(np.float32), arms


def evaluate_patient(task):
    root, patient, pairs, truth_identity, options, use_sam = task
    receipt = read_json(score_path(root, "validation", patient))
    r3d = {e["code"]: e for e in receipt["pairs"]}
    sam = {}
    if use_sam:
        sreceipt = read_json(sam_score_path(root, "validation", patient))
        sam = {e["code"]: e for e in sreceipt["pairs"]}
    truth, reference = load_mask(truth_identity)
    results = {}
    for pair in pairs:
        entry = r3d[pair.code]
        if entry["baseline"] != pair.mask:
            raise ValueError(f"{patient}/{pair.code}: score baseline differs")
        baseline, image = load_mask(pair.mask)
        box = entry["box"]
        prior, arms = load_arms(entry, sam.get(pair.code) if use_sam else None)
        base = crop(baseline, box)
        roi = D.region_of_interest(base, prior[0])
        window = Window.build(base, roi, prior, arms)
        local_truth = crop(truth, box)
        full_tp, full_den = counts(baseline, truth)
        win_tp, win_den = counts(base, local_truth)
        base_dice = dice_counts(full_tp, full_den)
        gains = np.zeros((len(options[pair.code]), 3), np.float64)
        for i, policy in enumerate(options[pair.code][1:], 1):
            candidate = window.apply(policy)
            cand_tp, cand_den = counts(candidate, local_truth)
            gains[i] = dice_counts(full_tp - win_tp + cand_tp, full_den - win_den + cand_den) - base_dice
        # Oracle: best legal adjacent migration inside the ROI under GT.
        oracle = roi_delete_oracle(base, local_truth, roi, full_tp, full_den, win_tp, win_den, base_dice)
        results[pair.code] = (base_dice, gains, oracle)
    return patient, results


def roi_delete_oracle(base, truth, roi, full_tp, full_den, win_tp, win_den, base_dice):
    best = base.copy()
    for h in range(3):
        removable = roi & (base == h + 1) & (truth <= h)
        best[removable] = h
    tp, den = counts(best, truth)
    return dice_counts(full_tp - win_tp + tp, full_den - win_den + den) - base_dice


def bootstrap(deltas, seed=0):
    rng = np.random.default_rng(seed)
    deltas = np.asarray(deltas)
    n = len(deltas)
    mean = deltas[rng.integers(0, n, (10000, n))].mean(1)
    low, high = np.percentile(mean, (2.5, 97.5))
    positive = int((deltas > 1e-8).sum())
    negative = int((deltas < -1e-8).sum())
    return {"mean_gain_pp": float(deltas.mean()), "ci95_pp": [float(low), float(high)],
            "improved": positive, "worsened": negative, "unchanged": n - positive - negative,
            "patients": n, "pass": bool(deltas.mean() > 0.01 and positive > negative)}


def objective(values, policy):
    means = values.mean(0)
    if any(means[h] <= 0.01 for h in policy.heads):
        return -np.inf
    return float(means.sum())


def fit_policy(values, options, allowed_arms, seed):
    groups = defaultdict(list)
    for i, policy in enumerate(options):
        if policy.arm in allowed_arms:
            groups[policy.family].append(i)
    if len(groups) <= 1:
        return 0

    def best(ids, y):
        return max(ids, key=lambda i: objective(y[:, i], options[i]))

    indices = list(groups.values())
    evaluation = np.zeros((len(indices), len(values), 3), np.float64)
    for inner_train, inner_test in KFold(5, shuffle=True, random_state=seed).split(values):
        for j, group in enumerate(indices):
            chosen = best(group, values[inner_train])
            if np.isfinite(objective(values[inner_train, chosen], options[chosen])):
                evaluation[j, inner_test] = values[inner_test, chosen]
    winning = max(range(len(indices)), key=lambda j: objective(evaluation[j], options[indices[j][0]]))
    selected = best(indices[winning], values)
    return selected if np.isfinite(objective(values[:, selected], options[selected])) else 0


def nested(table, options, allowed_arms):
    values = np.asarray(table)
    gains = np.zeros((len(values), 3), np.float64)
    choices = []
    for fold, (train, heldout) in enumerate(KFold(5, shuffle=True, random_state=20261125).split(values)):
        selected = fit_policy(values[train], options, allowed_arms, 10 + fold)
        gains[heldout] = values[heldout, selected]
        choices.append({"fold": fold, "policy": options[selected].to_json(),
                        "heldout_indices": heldout.tolist()})
    stats = [bootstrap(gains[:, h]) for h in range(3)]
    final = fit_policy(values, options, allowed_arms, 99)
    return stats, final, gains, choices


def score_path(output, split, case_id):
    return Path(output) / "r3d_scores" / split / f"{case_id}.json"


def sam_score_path(output, split, case_id):
    return Path(output) / "sam_scores" / split / f"{case_id}.json"


def has_sam(output):
    return (Path(output) / "sam_scores" / "validation").exists()


def write_csv(path, rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def freeze_arm_set(measured, ids, options, arm_sets):
    """Run nested selection under each arm-set; return per-set oof means and the per-code outcome."""
    outcomes = {}
    for name, arms in arm_sets.items():
        policies, per_region_oof = {}, []
        for code in range(1, 16):
            if not options[code][1:]:
                policies[code] = (Policy("P"), [None, None, None])
                continue
            table = np.stack([measured[p][code][1] for p in ids])
            stats, final, oof, _ = nested(table, options[code], arms)
            policies[code] = (options[code][final], stats)
            per_region_oof.append(oof.mean(0))
        mean = float(np.mean([v for row in per_region_oof for v in row])) if per_region_oof else 0.0
        outcomes[name] = {"policies": policies, "oof_mean_pp": mean}
    return outcomes


def choose_frozen(outcomes):
    full, pr = outcomes.get("PRS"), outcomes.get("PR")
    if full and pr and full["oof_mean_pp"] > pr["oof_mean_pp"]:
        return "PRS"
    return "PR" if pr else next(iter(outcomes))


def validate(args):
    root = args.output.resolve()
    frozen_path = root / "frozen3d.json"
    if frozen_path.exists():
        load_frozen(root)
        return
    completed = read_json(root / "training3d" / "completed.json")
    checkpoint = completed["checkpoint"]
    check_file(checkpoint)
    pairs = catalog(args.project, "validation")
    truth_rows = {r["case_id"]: r["gt"] for r in split_manifest(args.project, "validation", truth=True)}
    ids = sorted(pairs)
    if len(ids) != 80 or set(ids) != set(truth_rows):
        raise ValueError("Selection requires the full val80 cohort")
    use_sam = has_sam(root) and not args.no_sam
    arm_sets = {k: v for k, v in ARM_SETS.items() if use_sam or k != "PRS"}
    options = {code: candidates(heads_for(code, args.scope), ARMS if use_sam else ("P", "R"))
               for code in range(1, 16)}
    tasks = [(root, p, pairs[p], truth_rows[p], options, use_sam) for p in ids]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        measured = dict(pool.map(evaluate_patient, tasks, chunksize=1))
    outcomes = freeze_arm_set(measured, ids, options, arm_sets)
    chosen = choose_frozen(outcomes)
    policies = outcomes[chosen]["policies"]
    summary, per_case = [], []
    policy_json = {}
    for code in range(1, 16):
        policy, stats = policies[code]
        policy_json[str(code)] = policy.to_json()
        baseline = np.stack([measured[p][code][0] for p in ids])
        table = np.stack([measured[p][code][1] for p in ids]) if options[code][1:] else None
        oracle = np.stack([measured[p][code][2] for p in ids])
        final_index = options[code].index(policy) if policy in options[code] else 0
        for h, region in enumerate(HEADS):
            stat = stats[h] or {"mean_gain_pp": 0.0, "ci95_pp": [0.0, 0.0], "improved": 0,
                                "worsened": 0, "unchanged": len(ids), "patients": len(ids), "pass": False}
            summary.append({"mask_code": code, "region": region, "arm_set": chosen,
                            "enabled": h in policy.heads, "selected_arm": policy.arm,
                            "baseline_pct": float(baseline[:, h].mean()), **stat,
                            "roi_delete_oracle_pp": float(oracle[:, h].mean()),
                            "frozen_val80_gain_pp_descriptive":
                                float(table[:, final_index, h].mean()) if table is not None else 0.0})
            for j, case_id in enumerate(ids):
                per_case.append({"case_id": case_id, "mask_code": code, "region": region,
                                 "baseline_pct": float(baseline[j, h]),
                                 "roi_delete_oracle_pp": float(oracle[j, h]),
                                 "frozen_gain_pp_descriptive":
                                     float(table[j, final_index, h]) if table is not None else 0.0})
    write_csv(root / "val80_summary.csv", summary)
    write_csv(root / "val80_per_case.csv", per_case)
    write_json(root / "val80_summary.json", {
        "rows": summary, "patient_order": ids, "scope": args.scope, "use_sam": use_sam,
        "arm_set_oof_mean_pp": {k: v["oof_mean_pp"] for k, v in outcomes.items()},
        "frozen_arm_set": chosen, "selection": "outer 5 patient folds; inner 5 for arm+threshold family",
        "ci": "paired 95% percentile bootstrap, per region; unadjusted"})
    write_json(frozen_path, {"schema": SCHEMA, "checkpoint": checkpoint, "training_patients": 320,
                             "validation_patients": 80, "validation_ids": ids, "scope": args.scope,
                             "step": args.step, "mirror": not args.no_mirror, "use_sam": use_sam,
                             "frozen_arm_set": chosen,
                             "baseline_model": completed["baseline_model"],
                             "policy_source": policy_source(), "policies": policy_json,
                             "validation": file_identity(root / "val80_summary.json")})
    print(f"Frozen {sum(r['enabled'] for r in summary)}/45 units under arm-set {chosen}", flush=True)


def load_frozen(output):
    frozen = read_json(Path(output) / "frozen3d.json")
    if (frozen["schema"] != SCHEMA or frozen["policy_source"] != policy_source()
            or frozen["validation_patients"] != 80 or frozen["training_patients"] != 320):
        raise ValueError("Frozen policy is from different code or an incomplete cohort")
    check_file(frozen["checkpoint"])
    policies = {int(code): Policy.from_json(policy) for code, policy in frozen["policies"].items()}
    if set(policies) != set(range(1, 16)):
        raise ValueError("Frozen policy must cover all 15 combinations")
    return {**frozen, "policies": policies}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project", type=Path, default=D.PROJECT)
    p.add_argument("--output", type=Path, default=D.DEFAULT_OUTPUT)
    p.add_argument("--scope", choices=("all", "priority"), default="all")
    p.add_argument("--step", type=int, default=64)
    p.add_argument("--no-mirror", action="store_true")
    p.add_argument("--no-sam", action="store_true")
    p.add_argument("--workers", type=int, default=8)
    validate(p.parse_args())


if __name__ == "__main__":
    main()
