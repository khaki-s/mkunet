"""Nested patient-level validation of the refinement policies, followed by freezing."""
from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import csv
from pathlib import Path

import numpy as np
from sklearn.model_selection import KFold

from ..common import check_file, file_identity, read_json, same_grid, write_json
from .dataset import HEADS, PROJECT, catalog, heads_for, load_mask, split_manifest
from .infer import load_scores, score_path
from .policy import Policy, Window, candidates

SCHEMA = "brats_missing15_sam3_frozen_v1"


def policy_source():
    root = Path(__file__).parent
    return {n: file_identity(root / n)["sha256"] for n in ("policy.py", "select_val80.py", "infer.py")}


def dice_counts(tp, denominator):
    return np.divide(200. * tp, denominator, out=np.full(3, 100.), where=denominator != 0)


def counts(prediction, truth):
    tp = []
    denominator = []
    for h in range(3):
        pred, ref = prediction > h, truth > h
        tp.append(np.count_nonzero(pred & ref))
        denominator.append(np.count_nonzero(pred) + np.count_nonzero(ref))
    return np.asarray(tp), np.asarray(denominator)


def window_from_scores(labels, info):
    if not info["records"]:
        return None
    check_file(info["scores"])
    with np.load(info["scores"]["path"], allow_pickle=False) as file:
        return Window.build(labels, info["records"], file["score"], file["probability"])


def evaluate_patient(task):
    root, patient, pairs, truth_identity, checkpoint, rules = task
    predictions = load_scores(root, "validation", patient, checkpoint)
    if set(predictions) != {p.code for p in pairs}:
        raise ValueError(f"{patient}: validation predictions are incomplete")
    truth, reference = load_mask(truth_identity)
    results = {}
    threshold_control = 0.
    for pair in pairs:
        scores = predictions[pair.code]
        options = rules[pair.code]
        if scores["baseline"] != pair.mask:
            raise ValueError("Prediction has a different baseline")
        labels, image = load_mask(pair.mask)
        same_grid(image, reference)
        tp, den = counts(labels, truth)
        base = dice_counts(tp, den)
        gains = np.zeros((len(options), 3), np.float64)
        window = window_from_scores(labels, scores)
        local_deletion_oracle = np.zeros(3, np.float64)
        if window is not None:
            local_truth = truth[:, :, window.z]
            local_tp, local_den = counts(window.labels, local_truth)
            available = np.isfinite(window.model_probability)
            n_removable = np.array([
                np.count_nonzero(available[h] & (window.labels == h + 1) & (local_truth <= h))
                for h in range(3)
            ])
            local_deletion_oracle = dice_counts(tp, den - n_removable) - base
            if pair.code == 8:
                et = window.baseline_probability[2] >= np.float16(.30)
                comparison = window.labels.copy()
                comparison[available[2] & et & (window.labels == 2)] = 3
                comparison[available[2] & ~et & (window.labels == 3)] = 2
                comp_tp, comp_den = counts(comparison, local_truth)
                threshold_control = float(dice_counts(tp - local_tp + comp_tp, den - local_den + comp_den)[2] - base[2])
            for i, policy in enumerate(options[1:], 1):
                candidate = window.apply(policy)
                new_tp, new_den = counts(candidate, local_truth)
                gains[i] = dice_counts(tp - local_tp + new_tp, den - local_den + new_den) - base
        results[pair.code] = (base, gains, local_deletion_oracle)
    return patient, (results, threshold_control)


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
            "patients": n, "pass": bool(deltas.mean() > 0 and low > 0 and positive > negative)}


def objective(values, policy):
    means = values.mean(0)
    if any(means[h] <= 1e-10 for h in policy.heads):
        return -np.inf
    return float(means.sum())


def fit_policy(values, options, allowed, seed):
    """Inner OOF chooses an edit family, then refit its numeric thresholds."""
    groups = defaultdict(list)
    for i, policy in enumerate(options):
        if set(policy.heads).issubset(allowed):
            groups[policy.family].append(i)
    if len(groups) == 1:
        return 0

    def best(ids, y):
        return max(ids, key=lambda i: objective(y[:, i], options[i]))

    indices = list(groups.values())
    evaluation = np.zeros((len(indices), len(values), 3), dtype=np.float64)
    for inner_train, inner_test in KFold(5, shuffle=True, random_state=seed).split(values):
        for j, group in enumerate(indices):
            chosen = best(group, values[inner_train])
            if np.isfinite(objective(values[inner_train, chosen], options[chosen])):
                evaluation[j, inner_test] = values[inner_test, chosen]
    winning = max(range(len(indices)), key=lambda j: objective(evaluation[j], options[indices[j][0]]))
    selected = best(indices[winning], values)
    return selected if np.isfinite(objective(values[:, selected], options[selected])) else 0


def nested(table, options):
    values = np.asarray(table)
    allowed = {h for p in options for h in p.heads}
    gains = np.zeros((len(values), 3), dtype=np.float64)
    chosen = []
    for fold, (train, heldout) in enumerate(KFold(5, shuffle=True, random_state=20261125).split(values)):
        selected = fit_policy(values[train], options, allowed, 10 + fold)
        gains[heldout] = values[heldout, selected]
        chosen.append({"fold": fold, "policy": options[selected].to_json(),
                       "heldout_indices": heldout.tolist()})
    stats = [bootstrap(gains[:, h]) for h in range(3)]
    final = fit_policy(values, options, {h for h, stat in enumerate(stats) if stat["pass"]}, 99)
    return stats, final, gains, chosen


def write_csv(path, rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_frozen(output):
    frozen = read_json(Path(output) / "frozen.json")
    if (frozen["schema"] != SCHEMA or frozen["policy_source"] != policy_source()
            or frozen["validation_patients"] != 80 or frozen["training_patients"] != 320):
        raise ValueError("Expected the unmodified policy from full train320 and val80")
    check_file(frozen["checkpoint"])
    policies = {int(code): Policy.from_json(policy) for code, policy in frozen["policies"].items()}
    if set(policies) != set(range(1, 16)):
        raise ValueError("Frozen policy must contain all 15 combinations")
    return {**frozen, "policies": policies}


def validate(args):
    root = args.output.resolve()
    frozen_path = root / "frozen.json"
    if frozen_path.exists():
        load_frozen(root)
        return
    completed = read_json(root / "training" / "completed.json")
    train = completed["training"]
    checkpoint = completed["checkpoint"]
    check_file(checkpoint)
    pairs = catalog(args.project, "validation")
    truth_rows = {r["case_id"]: r["gt"] for r in split_manifest(args.project, "validation", truth=True)}
    ids = sorted(pairs)
    if len(ids) != 80 or set(ids) != set(truth_rows) or len(train["cases"]) != 320:
        raise ValueError("Formal selection requires full train320 and val80")
    if set(ids) & set(train["cases"]):
        raise ValueError("Training and validation patients overlap")
    if next(iter(pairs.values()))[0].baseline_model != train["baseline_model"]:
        raise ValueError("Validation and training baseline are different")
    # Bind the cache schedule that created the score receipts to the policy being frozen.
    for patient in ids:
        receipt = read_json(score_path(root, "validation", patient))
        binding = read_json(receipt["binding"]["packet"]["path"])["binding"]
        if (binding["scope"], binding["max_slices"], binding["split"]) != (args.scope, args.max_slices, "validation"):
            raise ValueError("Validation score schedule differs from the requested policy")
    options = {code: candidates(heads_for(code, args.scope)) for code in range(1, 16)}
    tasks = [(root, patient, pairs[patient], truth_rows[patient], checkpoint, options) for patient in ids]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        measured = dict(pool.map(evaluate_patient, tasks, chunksize=1))
    policies = {}
    summary = []
    per_case = []
    fold_choices = {}
    for code in range(1, 16):
        baseline, table, oracle = (np.stack(v) for v in zip(*(measured[patient][0][code] for patient in ids)))
        stats, final, oof, choices = nested(table, options[code])
        selected = options[code][final]
        policies[str(code)] = selected.to_json()
        fold_choices[str(code)] = choices
        for h, region in enumerate(HEADS):
            summary.append({"mask_code": code, "region": region, "enabled": h in selected.heads,
                            "baseline_pct": float(baseline[:, h].mean()), **stats[h],
                            "queried_slices_adjacent_delete_oracle_pp": float(oracle[:, h].mean()),
                            "frozen_val80_gain_pp_descriptive": float(table[:, final, h].mean())})
            for j, case_id in enumerate(ids):
                per_case.append({"case_id": case_id, "mask_code": code, "region": region,
                                 "baseline_pct": float(baseline[j, h]), "oof_gain_pp": float(oof[j, h]),
                                 "queried_slices_adjacent_delete_oracle_pp": float(oracle[j, h]),
                                 "frozen_gain_pp_descriptive": float(table[j, final, h])})
    write_csv(root / "val80_per_case.csv", per_case)
    write_csv(root / "val80_summary.csv", summary)
    write_json(root / "val80_summary.json", {"rows": summary, "fold_choices": fold_choices, "patient_order": ids,
                                             "ci": "paired 95% percentile bootstrap, per region; unadjusted",
                                             "mask08_et_same_slice_threshold_0.30": bootstrap([measured[p][1] for p in ids]),
                                             "selection": "outer 5 patient folds; inner 5 for edit family"})
    write_json(frozen_path, {"schema": SCHEMA, "checkpoint": checkpoint, "training_patients": 320,
                             "validation_patients": 80, "validation_ids": ids, "scope": args.scope,
                             "max_slices": args.max_slices, "baseline_model": train["baseline_model"],
                             "policy_source": policy_source(), "policies": policies,
                             "validation": file_identity(root / "val80_summary.json")})
    print(f"Frozen {sum(row['enabled'] for row in summary)}/45 units", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project", type=Path, default=PROJECT)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--scope", choices=("all", "priority"), default="priority")
    p.add_argument("--max-slices", type=int, default=5)
    p.add_argument("--workers", type=int, default=6)
    validate(p.parse_args())


if __name__ == "__main__":
    main()
