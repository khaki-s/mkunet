#!/usr/bin/env python
"""CPU integration checks; no downloaded models, patient scans, or network calls."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import nibabel as nib
import numpy as np
import torch

from nnunetv2.utilities.brats2021_audit_v2.interactive.schemas import PointPrompt, SpatialROI
from nnunetv2.utilities.brats2021_audit_v2.interactive.session import CorrectionSession
from nnunetv2.utilities.brats2021_audit_v2.interactive.controller import (
    RuleController, LLMController, AmbiguousInstruction, action_from_json,
)
from nnunetv2.utilities.brats2021_audit_v2.interactive.backends import NNInteractiveBackend


class ZeroCorrector(torch.nn.Module):
    def forward(self, x):
        return torch.zeros((x.shape[0], 3, *x.shape[2:]), device=x.device)


class FakeGenerator:
    def __init__(self, value):
        self.value = value

    def generate(self, system, user, *, images=()):
        assert "Negative ET" in system
        assert "selected_points" in user
        return self.value


class FakeNNInteractive:
    """Checks the actual public API names/order, not pretrained segmentation."""
    def __init__(self):
        self.calls = []

    def set_image(self, image):
        self.calls.append("image")
        assert image.ndim == 4 and image.shape[0] == 1

    def set_target_buffer(self, output):
        self.calls.append("buffer")
        self.output = output

    def add_initial_seg_interaction(self, mask, run_prediction=False):
        self.calls.append("initial")
        assert not run_prediction
        self.output.copy_(torch.from_numpy(mask))

    def add_point_interaction(self, coordinates, include_interaction, run_prediction):
        self.calls.append(("point", coordinates, run_prediction))
        self.output[coordinates] = int(include_interaction)


def expect_error(fn):
    try:
        fn()
    except (ValueError, FileNotFoundError):
        return
    raise AssertionError("Invalid request was accepted")


def main():
    torch.set_num_threads(2)
    torch.manual_seed(3)
    shape = (12, 16, 20)
    affine = np.diag([1., 2., 3., 1.])
    mri = torch.randn((4, *shape))
    logits = torch.ones((3, *shape))
    logits[0] *= 3
    logits[1] *= 2
    session = CorrectionSession(mri, [1,1,1,1], logits, affine, ZeroCorrector())
    session.register_roi("roi_1", SpatialROI((2,2,2), (10,14,18)))
    session.register_roi("roi_other", SpatialROI((1,1,1), (11,15,19)))
    session.register_point(PointPrompt("click_1", (5,6,7), "ET", False))
    action = RuleController().propose("框内 ET 分多了，保留周围水肿", session,
                                      roi_id="roi_1", point_ids=["click_1"])
    assert action.operation == "exclude" and action.replacement_class is None
    expect_error(lambda: RuleController().propose("这里不对", session, roi_id="roi_1"))
    raw = json.dumps(action.to_dict())
    assert LLMController(FakeGenerator(raw)).propose(
        "ET分多了", session, roi_id="roi_1", point_ids=["click_1"]) == action
    for payload in [
        {**action.to_dict(), "coordinates": [5,6,7]},
        {**action.to_dict(), "negative_point_ids": ["invented"]},
        {**action.to_dict(), "replacement_class": "BG"},
        {**action.to_dict(), "target": "TC"},
    ]:
        expect_error(lambda p=payload: action_from_json(json.dumps(p), session))
    expect_error(lambda: action_from_json('{"target":"ET","target":"WT"}', session))
    session.register_point(PointPrompt("tc_positive", (5,6,8), "TC", True))
    replacement = RuleController().propose("TC改成ET", session, roi_id="roi_1",
                                           point_ids=["tc_positive"])
    assert replacement.target == "TC" and replacement.replacement_class == "ET"
    changed_roi = {**action.to_dict(), "roi_id": "roi_other"}
    expect_error(lambda: LLMController(FakeGenerator(json.dumps(changed_roi))).propose(
        "ET分多了", session, roi_id="roi_1", point_ids=["click_1"]))
    before = session.logits.clone()
    report = session.apply_action(action)
    assert torch.equal(session.logits[:2], before[:2]), "ET negative changed WT/TC"
    assert session.logits[2,5,6,7] < 0
    assert report["prompt_satisfaction_before_constraints"] == 0.
    assert report["prompt_satisfaction_after_constraints"] == 1.
    assert session.undo() and torch.equal(session.logits, before)
    engine = FakeNNInteractive()
    backend = NNInteractiveBackend(None, mri[0:1].numpy(), affine,
                                   modality_index=0, inference_session=engine)
    session.corrector = backend
    session.apply_action(action)
    assert engine.calls[:3] == ["image","buffer","initial"]
    assert engine.calls[-1] == ("point", (3,4,5), True)
    assert session.logits[2,5,6,7] < 0 and torch.equal(session.logits[:2], before[:2])

    folder = Path(tempfile.mkdtemp(prefix="brats_prompt_cli_"))
    (folder/"splits.json").write_text(json.dumps([{"train":["synthetic_cli"],"val":["other_case"]}]))
    image = mri[0].numpy()
    nib.save(nib.Nifti1Image(image, affine), str(folder/"t1.nii.gz"))
    probs = torch.sigmoid(logits).numpy()
    np.savez_compressed(folder/"prediction.npz", probabilities=probs.transpose(0,3,2,1))
    scripts = Path(__file__).resolve().parent
    subprocess.run([sys.executable, str(scripts/"prepare_case.py"), "--t1", str(folder/"t1.nii.gz"),
                    "--probabilities", str(folder/"prediction.npz"), "--probability-axis-order", "zyx",
                    "--case-id","synthetic_cli","--split","train","--splits-file",str(folder/"splits.json"),
                    "--output",str(folder/"case.npz")], check=True)
    rejected = subprocess.run([
        sys.executable, str(scripts/"prepare_case.py"), "--t1", str(folder/"t1.nii.gz"),
        "--probabilities", str(folder/"prediction.npz"), "--probability-axis-order", "zyx",
        "--case-id","other_case","--split","train","--splits-file",str(folder/"splits.json"),
        "--output",str(folder/"bad_train.npz")], capture_output=True, text=True)
    assert rejected.returncode != 0 and not (folder/"bad_train.npz").exists()
    with np.load(folder/"case.npz") as data:
        assert data["mri"].shape == (4,*shape)
        assert np.allclose(1/(1+np.exp(-data["logits"])), probs)
        assert not data["mri"][1:].any()
    interactions = {
        "rois":{"roi_1":{"start":[2,2,2],"stop":[10,14,18]}},
        "points":[{"point_id":"click_1","voxel":[5,6,7],"target":"ET","positive":False}],
        "commands":[{"text":"ET分多了，保留周围水肿", "roi_id":"roi_1","point_ids":["click_1"]}],
    }
    (folder/"interactions.json").write_text(json.dumps(interactions, ensure_ascii=False))
    subprocess.run([sys.executable,str(scripts/"correct_case.py"),"--case",str(folder/"case.npz"),
                    "--interactions",str(folder/"interactions.json"),"--output",str(folder/"corrected"),
                    "--demo-untrained","--accept-feedback"],check=True)
    result = nib.load(str(folder/"corrected/corrected_internal_labels.nii.gz"))
    assert np.allclose(result.affine, affine)
    assert result.get_fdata()[5,6,7] == 2, "ET negative should retain NCR instead of BG"
    replay = json.loads((folder/"corrected/replay.json").read_text())
    replay["commands"].append({"undo": True})
    (folder/"undo.json").write_text(json.dumps(replay))
    subprocess.run([sys.executable,str(scripts/"correct_case.py"),"--case",str(folder/"case.npz"),
                    "--interactions",str(folder/"undo.json"),"--output",str(folder/"undone"),
                    "--demo-untrained"], check=True)
    with np.load(folder/"undone/state.npz") as restored, np.load(folder/"case.npz") as original:
        assert np.array_equal(restored["logits"], original["logits"])
    wrong_replay = subprocess.run([
        sys.executable,str(scripts/"correct_case.py"),"--case",str(folder/"corrected/state.npz"),
        "--interactions",str(folder/"undo.json"),"--output",str(folder/"bad_replay"),
        "--demo-untrained"], capture_output=True, text=True)
    assert wrong_replay.returncode != 0 and not (folder/"bad_replay").exists()
    result = {
        "status":"passed", "rule_and_stub_llm_validation":True, "official_backend_api_stub":True,
        "real_prepare_and_correction_cli":True, "probability_axis_conversion":True,
        "cross_process_replay_and_undo":True, "temporary_artifacts":str(folder),
        "replay_wrong_case_rejected":True, "train_whitelist_checked":True,
        "limits":"No real Qwen/nnInteractive weights or quality evaluation; untrained corrector only tests plumbing.",
    }
    print(json.dumps(result, indent=2))
    if len(sys.argv)>1:
        Path(sys.argv[1]).write_text(json.dumps(result,indent=2)+"\n")


if __name__ == "__main__":
    main()
