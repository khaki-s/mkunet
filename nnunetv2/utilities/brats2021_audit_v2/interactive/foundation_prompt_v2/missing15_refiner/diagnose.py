"""Post-hoc diagnostics for a finished refiner run (val80 GT + train320 GT only; never test).

signal : on the val80 queried planes, does the refined SAM score separate baseline errors
         better than the baseline probability itself?  Per (mask_code, region):
         - inside the baseline region, AUROC for "voxel is a false positive" using 1-score vs 1-prob
         - outside (prob >= PROB_FLOOR candidates), AUROC for "voxel is a missed tumour voxel"
         - mean |score - prob| (does the decoder move away from its prior at all?)
shift  : fold_0 in-sample train320 baseline vs val80 baseline, Dice and FP/FN volume per unit.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from ..common import check_file, read_json, write_json
from .dataset import HEADS, PROB_FLOOR, PROJECT, catalog, load_mask, split_manifest
from .infer import score_path

SAMPLE = 4000   # voxels per class per (patient, pair, region) to bound memory


def _take(rng, values, n=SAMPLE):
    return values if len(values) <= n else values[rng.choice(len(values), n, replace=False)]


def signal_patient(task):
    root, patient, truth_identity = task
    rng = np.random.default_rng(abs(hash(patient)) % 2**32)
    truth, _ = load_mask(truth_identity)
    receipt = read_json(score_path(root, "validation", patient))
    packet = {g["code"]: g for g in read_json(receipt["binding"]["packet"]["path"])["pairs"]}
    out = []
    for group in receipt["pairs"]:
        if not group["records"]:
            continue
        check_file(group["scores"])
        with np.load(group["scores"]["path"]) as z:
            score, prob = z["score"].astype(np.float32), z["probability"].astype(np.float32)
        with np.load(packet[group["code"]]["slices"]["path"]) as z:
            labels = z["labels"]
        # score rows follow the record order of the packet, see infer.score_patient
        for i, record in enumerate(group["records"]):
            h, zz = record["head"], record["z"]
            base = labels[i] > h
            gt = truth[:, :, zz] > h
            s, p = score[i], prob[i]
            inside_fp, inside_tp = base & ~gt, base & gt
            cand = ~base & (p >= PROB_FLOOR)
            if h:
                cand &= labels[i] >= h
            out.append({"code": group["code"], "head": h, "patient": patient,
                        "in_fp_s": _take(rng, s[inside_fp]), "in_fp_p": None, "in_tp_s": _take(rng, s[inside_tp]),
                        "n_fp": int(inside_fp.sum()), "n_tp": int(inside_tp.sum()),
                        "fp_vals": (_take(rng, np.stack([s[inside_fp], p[inside_fp]], 1))),
                        "tp_vals": (_take(rng, np.stack([s[inside_tp], p[inside_tp]], 1))),
                        "c_pos": _take(rng, np.stack([s[cand & gt], p[cand & gt]], 1)),
                        "c_neg": _take(rng, np.stack([s[cand & ~gt], p[cand & ~gt]], 1)),
                        "move": float(np.abs(s - p)[base | cand].mean()) if (base | cand).any() else 0.})
    return out


def _auc(pos, neg, col, sign):
    if len(pos) < 20 or len(neg) < 20:
        return float("nan")
    y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
    return float(roc_auc_score(y, sign * np.r_[pos[:, col], neg[:, col]]))


def signal(args):
    root = args.output.resolve()
    pairs = catalog(args.project, "validation")
    truth = {r["case_id"]: r["gt"] for r in split_manifest(args.project, "validation", truth=True)}
    tasks = [(root, cid, truth[cid]) for cid in sorted(pairs)]
    with ProcessPoolExecutor(args.workers) as pool:
        items = [x for chunk in pool.map(signal_patient, tasks, chunksize=1) for x in chunk]
    rows = []
    for code in sorted({x["code"] for x in items}):
        for h in sorted({x["head"] for x in items if x["code"] == code}):
            sel = [x for x in items if x["code"] == code and x["head"] == h]
            fp = np.concatenate([x["fp_vals"] for x in sel]) if sel else np.zeros((0, 2))
            tp = np.concatenate([x["tp_vals"] for x in sel]) if sel else np.zeros((0, 2))
            cp = np.concatenate([x["c_pos"] for x in sel])
            cn = np.concatenate([x["c_neg"] for x in sel])
            rows.append({"mask_code": code, "region": HEADS[h], "planes": len(sel),
                         "fp_voxels": sum(x["n_fp"] for x in sel), "tp_voxels": sum(x["n_tp"] for x in sel),
                         # higher = better at flagging false positives (low score/prob means "delete")
                         "delete_auc_sam": _auc(fp, tp, 0, -1), "delete_auc_prob": _auc(fp, tp, 1, -1),
                         "add_auc_sam": _auc(cp, cn, 0, 1), "add_auc_prob": _auc(cp, cn, 1, 1),
                         "mean_abs_score_minus_prob": float(np.mean([x["move"] for x in sel]))})
    _write(root / "diagnostics" / "val80_signal.csv", rows)
    return rows


def _logit(x):
    x = np.clip(x, 1e-4, 1 - 1e-4)
    return np.log(x / (1 - x))


def _fused_auc(blocks, labels, groups):
    """Patient-grouped 5-fold logistic fusion of [logit sam, logit prob, product]."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    X = np.stack([_logit(blocks[:, 0]), _logit(blocks[:, 1]), _logit(blocks[:, 0]) * _logit(blocks[:, 1])], 1)
    oof = np.zeros(len(labels))
    for tr, te in GroupKFold(5).split(X, labels, groups):
        oof[te] = LogisticRegression(max_iter=500).fit(X[tr], labels[tr]).decision_function(X[te])
    return float(roc_auc_score(labels, oof))


def fusion(args):
    """Headroom check: does a learned SAM+prior combination rank errors better than either alone?"""
    root = args.output.resolve()
    pairs = catalog(args.project, "validation")
    truth = {r["case_id"]: r["gt"] for r in split_manifest(args.project, "validation", truth=True)}
    tasks = [(root, cid, truth[cid]) for cid in sorted(pairs)]
    with ProcessPoolExecutor(args.workers) as pool:
        items = [x for chunk in pool.map(signal_patient, tasks, chunksize=1) for x in chunk]
    rows = []
    for code in sorted({x["code"] for x in items}):
        for h in sorted({x["head"] for x in items if x["code"] == code}):
            sel = [x for x in items if x["code"] == code and x["head"] == h]
            row = {"mask_code": code, "region": HEADS[h]}
            for side, pos_key, neg_key in (("delete", "tp_vals", "fp_vals"), ("add", "c_pos", "c_neg")):
                pos = [(x[pos_key], x["patient"]) for x in sel if len(x[pos_key])]
                neg = [(x[neg_key], x["patient"]) for x in sel if len(x[neg_key])]
                if not pos or not neg:
                    row.update({f"{side}_prob": np.nan, f"{side}_sam": np.nan, f"{side}_fused": np.nan})
                    continue
                P = np.concatenate([v for v, _ in pos]); N = np.concatenate([v for v, _ in neg])
                g = np.r_[np.concatenate([[p] * len(v) for v, p in pos]), np.concatenate([[p] * len(v) for v, p in neg])]
                y = np.r_[np.ones(len(P)), np.zeros(len(N))]
                B = np.concatenate([P, N])
                # label 1 = voxel truly belongs to the region (keep inside / add outside)
                row[f"{side}_prob"] = float(roc_auc_score(y, B[:, 1]))
                row[f"{side}_sam"] = float(roc_auc_score(y, B[:, 0]))
                row[f"{side}_fused"] = _fused_auc(B, y, g)
            rows.append(row)
    _write(root / "diagnostics" / "val80_fusion.csv", rows)
    return rows


def shift_one(task):
    split, code, mask_identity, truth_identity = task
    labels, _ = load_mask(mask_identity)
    truth, _ = load_mask(truth_identity)
    rec = {"split": split, "mask_code": code}
    for h, name in enumerate(HEADS):
        b, g = labels > h, truth > h
        s = int(b.sum()) + int(g.sum())
        rec[f"{name}_dice"] = 100. if s == 0 else 200. * int((b & g).sum()) / s
        rec[f"{name}_fp"] = int((b & ~g).sum())
        rec[f"{name}_fn"] = int((~b & g).sum())
    return rec


def shift(args):
    root = args.output.resolve()
    tasks = []
    for split in ("train", "validation"):
        truth = {r["case_id"]: r["gt"] for r in split_manifest(args.project, split, truth=True)}
        for cid, plist in catalog(args.project, split).items():
            tasks += [(split, p.code, p.mask, truth[cid]) for p in plist]
    with ProcessPoolExecutor(args.workers) as pool:
        recs = list(pool.map(shift_one, tasks, chunksize=4))
    rows = []
    for code in range(1, 16):
        for name in HEADS:
            row = {"mask_code": code, "region": name}
            for split, tag in (("train", "train"), ("validation", "val")):
                sel = [r for r in recs if r["split"] == split and r["mask_code"] == code]
                row[f"{tag}_n"] = len(sel)
                row[f"{tag}_dice"] = float(np.mean([r[f"{name}_dice"] for r in sel]))
                row[f"{tag}_fp_mean"] = float(np.mean([r[f"{name}_fp"] for r in sel]))
                row[f"{tag}_fn_mean"] = float(np.mean([r[f"{name}_fn"] for r in sel]))
            row["dice_gap_val_minus_train"] = row["val_dice"] - row["train_dice"]
            rows.append(row)
    _write(root / "diagnostics" / "baseline_shift.csv", rows)
    return rows


def _write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    write_json(path.with_suffix(".json"), rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("signal", "shift", "fusion"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--project", type=Path, default=PROJECT)
    p.add_argument("--workers", type=int, default=24)
    a = p.parse_args()
    rows = {"signal": signal, "shift": shift, "fusion": fusion}[a.stage](a)
    import pandas as pd
    pd.set_option("display.width", 250)
    print(pd.DataFrame(rows).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
