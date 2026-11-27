"""Optional SAM3 evidence arm (RS) and its no-SAM control (RS0).

SAM only speaks where R3D is most ambiguous. For each (case, combination, region)
the ambiguity a(z) = sum r_h(1-r_h) + sum |r_h - q_h| over the ROI picks the top-K
axial planes. A frozen SAM3 image encoder (Q/V LoRA on the decoder) gives a slice
logit; a 25-channel 2D adapter turns it, together with R3D and baseline context,
into a residual delta on logit(r_h). RS0 shares the adapter but zeroes the SAM
logit channel and loads no LoRA, so RS-minus-RS0 is the SAM evidence increment.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from ..dataset import channel as region_channel
from . import data3d as D

REGIONS = ("WT", "TC", "ET")
ADAPTER_CHANNELS = 25


class DilatedResidualBlock(nn.Module):
    def __init__(self, width, dilation):
        super().__init__()
        self.conv1 = nn.Conv2d(width, width, 3, padding=dilation, dilation=dilation)
        self.conv2 = nn.Conv2d(width, width, 3, padding=dilation, dilation=dilation)
        self.act = nn.LeakyReLU(0.01, inplace=True)

    def forward(self, x):
        y = self.act(self.conv1(x))
        y = self.conv2(y)
        return self.act(x + y)


class Adapter2D(nn.Module):
    """25->48 stem, six dilated residual blocks (d=1,2,4,4,2,1), zero-init 1x1 output."""

    def __init__(self, in_channels=ADAPTER_CHANNELS, width=48):
        super().__init__()
        self.stem = nn.Conv2d(in_channels, width, 3, padding=1)
        self.act = nn.LeakyReLU(0.01, inplace=True)
        self.blocks = nn.ModuleList(DilatedResidualBlock(width, d) for d in (1, 2, 4, 4, 2, 1))
        self.out = nn.Conv2d(width, 1, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, context):
        x = self.act(self.stem(context))
        for block in self.blocks:
            x = block(x)
        return self.out(x)


def ambiguity(region, prior_region, roi):
    """Per-plane ambiguity a(z) over the ROI from R3D region and baseline prior volumes."""
    uncertainty = (region * (1 - region)).sum(0) + np.abs(region - prior_region).sum(0)
    masked = np.where(roi, uncertainty, 0.0)
    return masked.sum(axis=(0, 1))


def select_planes(region, prior_region, roi, k, min_gap=2):
    """Top-K axial planes by ambiguity with a minimum spacing, spread by descending score."""
    scores = ambiguity(region, prior_region, roi)
    order = np.argsort(-scores, kind="stable")
    chosen = []
    for z in order:
        if scores[z] <= 0:
            break
        if all(abs(z - c) >= min_gap for c in chosen):
            chosen.append(int(z))
        if len(chosen) >= k:
            break
    return sorted(chosen)


def logit(values, eps=1e-4):
    v = np.clip(values, eps, 1 - eps)
    return np.log(v / (1 - v)).astype(np.float32)


def assemble_context(sam_logit, head, z, region_full, prior_full, prob_full,
                     base_full, mri_full, availability, enable_sam):
    """Build the 25-channel adapter input for region ``head`` on native plane ``z``.

    All *_full arrays are native (C, X, Y, Z) or (X, Y, Z); the SAM logit is (X, Y).
    Channels: SAM logit 1, r 3, p 3, p_h at z-1/z+1 2, |r_h-q_h| 1, baseline one-hot 4,
    available MRI 4, availability 4, region one-hot 3 = 25.
    """
    nz = region_full.shape[-1]
    zm, zp = max(z - 1, 0), min(z + 1, nz - 1)
    r = region_full[:, :, :, z]
    q = prior_full[:, :, :, z]
    p = prob_full[:, :, :, z]
    sam_plane = sam_logit if enable_sam else np.zeros_like(sam_logit)
    region_onehot = np.zeros((3, *sam_logit.shape), np.float32)
    region_onehot[head] = 1.0
    avail_planes = np.broadcast_to(np.asarray(availability, np.float32)[:, None, None], (4, *sam_logit.shape))
    channels = [
        sam_plane[None],
        r,
        p,
        prob_full[head, :, :, zm][None], prob_full[head, :, :, zp][None],
        np.abs(r[head] - q[head])[None],
        D.one_hot(base_full[:, :, z]),
        mri_full[:, :, :, z],
        avail_planes,
        region_onehot,
    ]
    return np.concatenate(channels, axis=0).astype(np.float32)


# --- native canvases from a cached window --------------------------------

from pathlib import Path  # noqa: E402

import nibabel as nib  # noqa: E402

from ...common import check_file, file_identity, read_json, same_grid, write_json  # noqa: E402
from ..dataset import MODALITIES, load_mask, load_probability  # noqa: E402
from ...sam_domain_data import FEATURE_KEYS, read_features, save_features, slice_rgb  # noqa: E402
from ...sam_domain_model import FrozenSamDomainEncoder, SAM_CHECKPOINT, SAM_SOURCE, source_identity  # noqa: E402
from .model3d import restore as restore_r3d  # noqa: E402
from .infer3d import sliding_window  # noqa: E402

SCHEMA = "brats_missing15_sam_arm_v2"


def native_region(box, region_window, native_prior):
    """Scatter the window R3D region volume onto a native prior canvas (outside the window = prior)."""
    full_region = native_prior.copy()
    region_view = (slice(None),) + tuple(slice(a, b) for a, b in box)
    full_region[region_view] = region_window
    return full_region


def native_arrays(pair):
    """Native baseline probability, nested region prior, labels, and z-scored MRI for a pair."""
    labels, reference = load_mask(pair.mask)
    probability = load_probability(pair, labels.shape)              # (3, X, Y, Z)
    prior_region = D.nested_projection_region(probability)          # (3, X, Y, Z) q_WT/q_TC/q_ET
    volume = np.zeros((4, *labels.shape), np.float32)
    for i, name in enumerate(MODALITIES):
        if D.availability(pair.code)[i]:
            identity = pair.modalities[name]
            check_file(identity)
            image = nib.load(identity["path"])
            same_grid(reference, image)
            volume[i] = image.get_fdata(dtype=np.float32)
    mri = D.zscore_mri(volume, D.availability(pair.code))
    return probability, prior_region, labels, mri, reference


from ...sam_domain_model import build_domain_decoder, sam_supervised_loss  # noqa: E402
from ..prompts import slice_prompt  # noqa: E402


class SAMArm(nn.Module):
    """SAM3 Q/V LoRA decoder plus two independent adapters: RS (with SAM) and RS0 (control)."""

    def __init__(self, source, checkpoint, lora_rank=8, device="cpu"):
        super().__init__()
        self.decoder = build_domain_decoder(source=source, checkpoint=checkpoint,
                                             mode="qv_lora", lora_rank=lora_rank, device=device)
        self.adapter_rs = Adapter2D().to(device)
        self.adapter_rs0 = Adapter2D().to(device)
        self.source = str(source)

    @property
    def mask_input_size(self):
        return self.decoder.mask_input_size

    def sam_logit(self, features, prompt, native_size):
        return self.decoder.forward_cached(features, prompt, native_size)[0, 0]

    def configuration(self):
        return {"schema": SCHEMA, "source": self.source,
                "decoder": self.decoder.configuration(), "adapter_channels": ADAPTER_CHANNELS}


def build_prompt(region_head_plane, decoder):
    """GT-free SAM prompt for one region plane from the R3D region map (binary + dense prior)."""
    values = slice_prompt(region_head_plane >= 0.5, region_head_plane)
    prompts = {k: torch.from_numpy(values[k][None]) for k in ("points", "point_labels", "box")}
    dense = torch.from_numpy(values["dense"])[None, None]
    prompts["mask_input"] = F.interpolate(dense, size=decoder.mask_input_size, mode="bilinear",
                                           align_corners=False)
    return prompts


def context_tensor(context, device):
    return torch.from_numpy(context[None]).to(device, dtype=torch.float32)


def arm_scores(model, features, prompt, native_size, head, z, region_full, prior_full,
               prob_full, base_full, mri_full, availability, device):
    """Return (rs_plane, rs0_plane) region probabilities for head ``head`` on plane ``z``."""
    sam_logit_plane = model.sam_logit(features, prompt, native_size).detach().cpu().numpy()
    r_head = region_full[head, :, :, z]
    base_logit = torch.from_numpy(logit(r_head)).to(device)
    rs0_context = context_tensor(assemble_context(
        np.zeros_like(sam_logit_plane), head, z, region_full, prior_full, prob_full,
        base_full, mri_full, availability, enable_sam=False), device)
    rs_context = context_tensor(assemble_context(
        sam_logit_plane, head, z, region_full, prior_full, prob_full,
        base_full, mri_full, availability, enable_sam=True), device)
    rs = torch.sigmoid(base_logit + model.adapter_rs(rs_context)[0, 0])
    rs0 = torch.sigmoid(base_logit + model.adapter_rs0(rs0_context)[0, 0])
    return rs.detach().cpu().numpy(), rs0.detach().cpu().numpy()


# --- shared preparation ---------------------------------------------------

import argparse  # noqa: E402

from ..dataset import catalog, heads_for, split_manifest  # noqa: E402


def region_volume(entry, r3d_model, device, patch, step, mirror):
    """Run the R3D model over one cached window to get its region score volume."""
    image = np.asarray(D.load_window(entry, "image"), np.float32)
    prob = np.asarray(D.load_window(entry, "prob"), np.float32)
    base = np.asarray(D.load_window(entry, "base"), np.int64)
    features = D.assemble_input(image, prob, base, entry["availability"])
    prior = D.prior_logits(prob)
    with torch.inference_mode():
        return sliding_window(r3d_model, features, prior, patch, step, device, mirror=mirror)


def prepare_pair(pair, entry, r3d_model, store, device, k, patch, step, mirror):
    """Native arrays, selected planes, and cached SAM features for one (case, combination)."""
    probability, prior_region, labels, mri, reference = native_arrays(pair)
    region_win = region_volume(entry, r3d_model, device, patch, step, mirror)
    box = entry["box"]
    full_region = native_region(box, region_win, prior_region)
    roi_full = D.region_of_interest(labels, probability[0])
    heads = heads_for(pair.code, "all")
    samples = []
    modality_cache = {}
    for head in heads:
        roi_win = roi_full[tuple(slice(a, b) for a, b in box)]
        planes = select_planes(region_win, D.nested_projection_region(prob_window(entry)),
                               roi_win, k)
        offset = box[2][0]
        for zw in planes:
            z = zw + offset
            modality = region_channel(head, pair.code)
            if modality not in modality_cache:
                identity = pair.modalities[MODALITIES[modality]]
                modality_cache[modality] = (identity, *modality_volume(identity, reference))
            identity, volume, ranges = modality_cache[modality]
            feature = store.get(pair.case_id, modality, z, identity, volume, ranges)
            samples.append({"head": head, "z": z, "modality": modality, "feature": feature})
    return {"probability": probability, "prior_region": prior_region, "labels": labels,
            "mri": mri, "full_region": full_region, "roi_full": roi_full,
            "availability": D.availability(pair.code), "box": box, "region_window": region_win,
            "samples": samples}


def prob_window(entry):
    return np.asarray(D.load_window(entry, "prob"), np.float32)


def plane_loss(s_logit, decoder_logit, target_plane, roi_plane, base_region_plane, source, device):
    """Weighted BCE + SoftDice on the adapter score, plus the official SAM supervision on the decoder."""
    target = torch.as_tensor(target_plane, device=device, dtype=torch.float32)
    roi = torch.as_tensor(roi_plane, device=device, dtype=torch.bool)
    base = torch.as_tensor(base_region_plane, device=device, dtype=torch.bool)
    error = roi & (target.bool() != base)
    weight = torch.zeros_like(target)
    weight[roi] = 1.0
    weight[error] = 3.0
    bce = F.binary_cross_entropy_with_logits(s_logit, target, reduction="none")
    bce = (bce * weight).sum() / weight.sum().clamp_min(1.0)
    prob = torch.sigmoid(s_logit)
    inter = (prob * target * roi).sum()
    dice = 1.0 - (2 * inter + 1e-5) / ((prob * roi).sum() + (target * roi).sum() + 1e-5)
    sam, _ = sam_supervised_loss(decoder_logit[None, None], target[None, None], source=source)
    return bce + 0.5 * dice + 0.2 * sam


def train(args):
    device = torch.device("cpu" if args.cpu else args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    root = args.output.resolve()
    completed = read_json(root / "training3d" / "completed.json")
    r3d_model, _ = restore_r3d(completed["checkpoint"], device)
    store = FeatureStore(root, args.source, args.checkpoint, args.device if not args.cpu else "cpu",
                         reuse=None if args.no_v1_reuse else args.v1_features)
    model = SAMArm(args.source, args.checkpoint, args.lora_rank, device)
    prepare = read_json(root / "prepare.json")
    binding = D.packet_binding("train", prepare["prepare_sha256"])
    cases = catalog(args.project, "train", per_case=args.per_case, limit_cases=args.limit_cases)
    truth = {r["case_id"]: r["gt"] for r in split_manifest(args.project, "train", truth=True)}
    lora = [p for p in model.decoder.parameters() if p.requires_grad]
    params = [{"params": lora, "lr": args.lora_lr},
              {"params": list(model.adapter_rs.parameters()) + list(model.adapter_rs0.parameters()),
               "lr": args.adapter_lr}]
    optimizer = torch.optim.AdamW(params, weight_decay=1e-4)
    output = root / "sam_training"
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    model.train()
    step = 0
    ids = sorted(cases)
    for epoch in range(args.epochs):
        for case_id in rng.permutation(ids):
            packet = D.read_packet(D.packet_path(root, "train", case_id), binding)
            by_code = {e["code"]: e for e in packet["pairs"]}
            for pair in cases[case_id]:
                entry = by_code[pair.code]
                prepared = prepare_pair(pair, entry, r3d_model, store, device,
                                        args.k, args.patch, args.step, not args.no_mirror)
                check_file(truth[case_id])
                gt = np.asanyarray(nib.load(truth[case_id]["path"]).dataobj).astype(np.int64)
                step = train_pair(model, optimizer, prepared, gt, device, step, args)
    atomic = output / "sam_arm.pth"
    temporary = atomic.with_suffix(".tmp")
    torch.save({"schema": SCHEMA, "configuration": model.configuration(),
                "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
                "lora_rank": args.lora_rank}, temporary)
    temporary.replace(atomic)
    write_json(output / "completed.json", {"checkpoint": file_identity(atomic),
               "r3d_checkpoint": completed["checkpoint"], "k": args.k, "steps": step})
    print(f"SAM arm trained {step} plane-steps; features {store.counts}; {atomic}", flush=True)


def train_pair(model, optimizer, prepared, gt, device, step, args):
    region, prior, prob = prepared["full_region"], prepared["prior_region"], prepared["probability"]
    base, mri, roi = prepared["labels"], prepared["mri"], prepared["roi_full"]
    avail = prepared["availability"]
    for sample in prepared["samples"]:
        head, z = sample["head"], sample["z"]
        features = read_features(sample["feature"], verify=False)
        r_head = region[head, :, :, z]
        prompt = build_prompt(r_head, model.decoder)
        sam_logit_plane = model.sam_logit(features, prompt, r_head.shape)
        base_logit = torch.from_numpy(logit(r_head)).to(device)
        rs_context = context_tensor(assemble_context(
            sam_logit_plane.detach().cpu().numpy(), head, z, region, prior, prob, base, mri,
            avail, enable_sam=True), device)
        rs0_context = context_tensor(assemble_context(
            np.zeros_like(r_head, np.float32), head, z, region, prior, prob, base, mri,
            avail, enable_sam=False), device)
        s_rs = base_logit + model.adapter_rs(rs_context)[0, 0]
        s_rs0 = base_logit + model.adapter_rs0(rs0_context)[0, 0]
        target_plane = (gt[:, :, z] > head).astype(np.float32)
        base_region_plane = base[:, :, z] > head
        roi_plane = roi[:, :, z]
        loss = (plane_loss(s_rs, sam_logit_plane, target_plane, roi_plane,
                           base_region_plane, model.source, device)
                + plane_loss(s_rs0, sam_logit_plane.detach(), target_plane, roi_plane,
                             base_region_plane, model.source, device))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        optimizer.step()
        step += 1
        if step % args.log_every == 0:
            print(f"sam step {step} loss={float(loss):.4f}", flush=True)
    return step


def restore_arm(identity, device):
    check_file(identity)
    state = torch.load(identity["path"], map_location="cpu", weights_only=True)
    if state["schema"] != SCHEMA:
        raise ValueError("SAM arm checkpoint schema mismatch")
    config = state["configuration"]
    model = SAMArm(config["source"], SAM_CHECKPOINT, state["lora_rank"], device)
    model.load_state_dict(state["state_dict"], strict=True)
    model.eval().requires_grad_(False)
    return model


def score(args):
    device = torch.device("cpu" if args.cpu else args.device)
    root = args.output.resolve()
    sam_completed = read_json(root / "sam_training" / "completed.json")
    model = restore_arm(sam_completed["checkpoint"], device)
    r3d_model, _ = restore_r3d(read_json(root / "training3d" / "completed.json")["checkpoint"], device)
    store = FeatureStore(root, args.source, args.checkpoint, args.device if not args.cpu else "cpu",
                         reuse=None if args.no_v1_reuse else args.v1_features)
    prepare = read_json(root / "prepare.json")
    binding = D.packet_binding(args.split, prepare["prepare_sha256"])
    codes = tuple(args.codes) if args.codes else tuple(range(1, 16))
    cases = catalog(args.project, args.split, codes=codes, limit_cases=args.limit_cases)
    ids = sorted(cases)[args.shard::args.shards]
    for index, case_id in enumerate(ids, 1):
        packet = D.read_packet(D.packet_path(root, args.split, case_id), binding)
        by_code = {e["code"]: e for e in packet["pairs"]}
        output = root / "sam_scores" / args.split / f"{case_id}.json"
        receipt_binding = {"sam_checkpoint": sam_completed["checkpoint"],
                           "r3d_checkpoint": read_json(root / "training3d" / "completed.json")["checkpoint"],
                           "k": args.k, "gt_used": False}
        if output.exists():
            if read_json(output)["binding"] != receipt_binding:
                raise ValueError(f"{output}: SAM scores belong to a different method")
            continue
        pairs = []
        for pair in cases[case_id]:
            entry = by_code[pair.code]
            prepared = prepare_pair(pair, entry, r3d_model, store, device,
                                    args.k, args.patch, args.step, not args.no_mirror)
            rs, rs0, covered = score_pair(model, prepared, entry, device)
            path = output.parent / case_id / f"mask_{pair.code:02d}.npz"
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(".npz.tmp")
            with temporary.open("wb") as handle:
                np.savez(handle, rs=rs.astype(np.float16), rs0=rs0.astype(np.float16),
                         covered=covered.astype(np.uint8))
            temporary.replace(path)
            pairs.append({"code": pair.code, "baseline": pair.mask, "box": entry["box"],
                          "scores": file_identity(path)})
        write_json(output, {"binding": receipt_binding, "pairs": pairs})
        print(f"{args.split} sam score {index}/{len(ids)} {case_id} features {store.counts}", flush=True)


def score_pair(model, prepared, entry, device):
    region_win = prepared["region_window"]
    rs, rs0 = region_win.copy(), region_win.copy()
    covered = np.zeros_like(region_win, np.uint8)
    region, prior, prob = prepared["full_region"], prepared["prior_region"], prepared["probability"]
    base, mri, avail = prepared["labels"], prepared["mri"], prepared["availability"]
    offset = entry["box"][2][0]
    with torch.inference_mode():
        for sample in prepared["samples"]:
            head, z = sample["head"], sample["z"]
            features = read_features(sample["feature"], verify=False)
            r_head = region[head, :, :, z]
            prompt = build_prompt(r_head, model.decoder)
            rs_plane, rs0_plane = arm_scores(model, features, prompt, r_head.shape, head, z,
                                             region, prior, prob, base, mri, avail, device)
            box = entry["box"]
            sel = (slice(box[0][0], box[0][1]), slice(box[1][0], box[1][1]))
            zw = z - offset
            rs[head, :, :, zw] = rs_plane[sel]
            rs0[head, :, :, zw] = rs0_plane[sel]
            covered[head, :, :, zw] = 1
    return rs, rs0, covered


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("train", "score"))
    p.add_argument("--project", type=Path, default=D.PROJECT)
    p.add_argument("--output", type=Path, default=D.DEFAULT_OUTPUT)
    p.add_argument("--split", choices=("validation", "test"), default="validation")
    p.add_argument("--codes", type=int, nargs="+", default=[])
    p.add_argument("--source", default=SAM_SOURCE)
    p.add_argument("--checkpoint", default=SAM_CHECKPOINT)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--lora-rank", type=int, default=8)
    p.add_argument("--lora-lr", type=float, default=1e-4)
    p.add_argument("--adapter-lr", type=float, default=3e-4)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--k", type=int, default=6)
    p.add_argument("--patch", type=int, default=128)
    p.add_argument("--step", type=int, default=64)
    p.add_argument("--per-case", type=int, default=6)
    p.add_argument("--limit-cases", type=int, default=0)
    p.add_argument("--seed", type=int, default=20261125)
    p.add_argument("--shards", type=int, default=1)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--v1-features", type=Path, default=V1_FEATURES)
    p.add_argument("--no-v1-reuse", action="store_true")
    p.add_argument("--no-mirror", action="store_true")
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()
    if args.stage == "score" and args.k == 6:
        args.k = 10
    if args.stage == "train":
        train(args)
    else:
        score(args)


if __name__ == "__main__":
    main()


def feature_cache_path(output, case_id, modality, z, sha):
    name = f"m{modality}_z{z:03d}_{sha[:16]}"
    return (Path(output) / "sam_features" / case_id / f"{name}.npz",
            Path(output) / "sam_features" / case_id / f"{name}.json")


V1_FEATURES = D.PROJECT / "sam_protocol" / "refiner" / "sam3_qv_priority_v1"


class FeatureStore:
    """SAM3 image features keyed by (case, modality, z, source sha).

    Lookup order: v2 cache, then the v1 cache (same encoder contract required),
    then a lazy encode. Only the remainder not found in v1 is ever encoded.
    """

    def __init__(self, output, source, checkpoint, device, reuse=V1_FEATURES):
        self.output, self.source, self.checkpoint, self.device = output, source, checkpoint, device
        self.encoder = None
        self.counts = {"v2": 0, "v1": 0, "encoded": 0}
        self.reuse = None
        if reuse:
            contract = Path(reuse) / "encoder.json"
            if contract.exists():
                value = read_json(contract)
                if (value["sam_checkpoint"]["path"] == str(Path(checkpoint).resolve())
                        and value["sam_source"] == str(Path(source).resolve())
                        and value["sources"] == source_identity(source)):
                    self.reuse = Path(reuse) / "features"
                else:
                    print(f"v1 feature cache {reuse} uses another encoder; not reused", flush=True)

    def _v1(self, case_id, modality, z, identity):
        if self.reuse is None:
            return None
        name = f"m{modality}_z{z:03d}_{identity['sha256'][:16]}.json"
        for split in ("train", "validation", "test"):
            receipt = self.reuse / split / case_id / name
            if receipt.exists():
                value = read_json(receipt)
                if value["input"] == identity:
                    check_file(value["file"])
                    return value["file"]
        return None

    def get(self, case_id, modality, z, identity, volume, ranges):
        npz, receipt = feature_cache_path(self.output, case_id, modality, z, identity["sha256"])
        if receipt.exists():
            value = read_json(receipt)
            check_file(value["file"])
            self.counts["v2"] += 1
            return value["file"]
        reused = self._v1(case_id, modality, z, identity)
        if reused is not None:
            self.counts["v1"] += 1
            return reused
        if self.encoder is None:
            self.encoder = FrozenSamDomainEncoder(self.source, self.checkpoint, self.device)
        features = self.encoder.encode(slice_rgb(volume[None], 0, z, [ranges]))
        npz.parent.mkdir(parents=True, exist_ok=True)
        file = save_features(npz, features)
        write_json(receipt, {"input": identity, "file": file})
        self.counts["encoded"] += 1
        return file


def modality_volume(identity, reference):
    check_file(identity)
    image = nib.load(identity["path"])
    same_grid(reference, image)
    volume = image.get_fdata(dtype=np.float32)
    nonzero = volume[volume != 0]
    lo, hi = np.percentile(nonzero, (1, 99)) if nonzero.size else (0.0, 1.0)
    return volume, (float(lo), max(float(hi - lo), 1e-6))
