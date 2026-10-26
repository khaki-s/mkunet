"""CPU regression tests for interactive semantics and the real trainable UNet."""
import json
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from nnunetv2.utilities.brats2021_audit_v2.interactive.feedback import load_feedback, save_feedback
from nnunetv2.utilities.brats2021_audit_v2.interactive.model import (
    CorrectorConfig, TorchPromptCorrector, load_corrector, save_corrector,
)
from nnunetv2.utilities.brats2021_audit_v2.interactive.prompts import gaussian_from_points, world_to_voxel
from nnunetv2.utilities.brats2021_audit_v2.interactive.schemas import (
    ATOMIC_REGIONS, REGIONS, PointPrompt, SpatialROI, StructuredAction,
)
from nnunetv2.utilities.brats2021_audit_v2.interactive.session import CorrectionSession
from nnunetv2.utilities.brats2021_audit_v2.interactive.training import (
    offline_training_step, simulate_error_click, simulate_training_example, sparse_region_bce,
    validate_simulation_identity, verify_training_split,
)


class ConstantDelta(nn.Module):
    def forward(self, data):
        return torch.full((data.shape[0], 3, *data.shape[2:]), -2.0, device=data.device)


class NonfiniteDelta(ConstantDelta):
    def forward(self, data):
        return super().forward(data) * float("nan")


def make_session(split="train", model=None):
    mri = torch.randn(4, 10, 10, 10)
    mri[1] = 0
    logits = torch.stack([torch.full((10, 10, 10), v) for v in (3.0, 2.0, 1.0)])
    session = CorrectionSession(mri, [1, 0, 1, 1], logits, np.diag([1, 2, 3, 1]),
                                model or ConstantDelta(), split=split)
    session.register_roi("r", SpatialROI((2, 2, 2), (8, 8, 8)))
    session.register_point(PointPrompt("negative_et", (4, 4, 4), "ET", False))
    return session


def negative_action():
    return StructuredAction("ET", "exclude", "r", negative_point_ids=("negative_et",))


class CoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.manual_seed(42)

    def test_negative_et_preserves_other_heads_roi_and_undo(self):
        session = make_session()
        before = session.logits.clone()
        record = session.apply_action(negative_action())
        self.assertTrue(torch.equal(session.logits[:2], before[:2]))
        outside = torch.ones(session.shape, dtype=torch.bool)
        outside[2:8, 2:8, 2:8] = False
        self.assertTrue(torch.equal(session.logits[:, outside], before[:, outside]))
        self.assertLess(float(session.logits[2, 4, 4, 4]), 0)
        self.assertEqual(int(session.confirmed_mask.sum()), 1)
        self.assertEqual(float(session.confirmed_labels[2, 4, 4, 4]), 0)
        self.assertEqual(record["prompt_satisfaction_after_constraints"], 1.0)
        self.assertTrue(session.undo())
        self.assertTrue(torch.equal(session.logits, before))
        self.assertFalse(session.confirmed_mask.any())
        self.assertFalse(session.undo())

    def test_explicit_ncr_replacement_supervises_three_heads(self):
        session = make_session()
        action = StructuredAction("ET", "replace", "r", negative_point_ids=("negative_et",),
                                  replacement_class="NCR")
        session.apply_action(action)
        self.assertEqual(session.confirmed_labels[:, 4, 4, 4].tolist(), [1, 1, 0])
        self.assertEqual(session.confirmed_mask[:, 4, 4, 4].tolist(), [True, True, True])
        self.assertTrue((session.logits[0] >= session.logits[1]).all())
        self.assertTrue((session.logits[1] >= session.logits[2]).all())

    def test_replacement_must_agree_with_all_registered_point_semantics(self):
        for head, target in enumerate(REGIONS):
            for positive in (False, True):
                for replacement, region_bits in ATOMIC_REGIONS.items():
                    with self.subTest(target=target, positive=positive, replacement=replacement):
                        session = make_session()
                        point = PointPrompt("replacement_point", (5, 5, 5), target, positive)
                        session.register_point(point)
                        action = StructuredAction(
                            target, "replace", "r",
                            positive_point_ids=(point.point_id,) if positive else (),
                            negative_point_ids=() if positive else (point.point_id,),
                            replacement_class=replacement,
                        )
                        if region_bits[head] != int(positive):
                            before = session.logits.clone()
                            with self.assertRaisesRegex(ValueError, "replacement_class conflicts"):
                                session.apply_action(action)
                            self.assertTrue(torch.equal(session.logits, before))
                            self.assertFalse(session.history)
                        else:
                            session.apply_action(action)
                            self.assertEqual(session.confirmed_labels[:, 5, 5, 5].tolist(), list(region_bits))

    def test_invalid_and_failed_backend_are_transactional(self):
        session = make_session()
        before = session.logits.clone()
        for action in (
            StructuredAction("ET", "exclude", "missing", negative_point_ids=("negative_et",)),
            StructuredAction("ET", "exclude", "r", negative_point_ids=("invented",)),
        ):
            with self.assertRaises(ValueError):
                session.apply_action(action)
        self.assertTrue(torch.equal(session.logits, before))
        self.assertFalse(session.history)
        session.corrector = NonfiniteDelta()
        with self.assertRaises(ValueError):
            session.apply_action(negative_action())
        self.assertTrue(torch.equal(session.logits, before))
        self.assertFalse(session.history)

    def test_contradictory_nested_points_rejected(self):
        session = make_session()
        session.register_point(PointPrompt("negative_wt", (5, 5, 5), "WT", False))
        session.apply_action(StructuredAction("WT", "exclude", "r", negative_point_ids=("negative_wt",)))
        before = session.logits.clone()
        session.register_point(PointPrompt("positive_et", (5, 5, 5), "ET", True))
        with self.assertRaisesRegex(ValueError, "Conflicting"):
            session.apply_action(StructuredAction("ET", "include", "r", positive_point_ids=("positive_et",)))
        self.assertTrue(torch.equal(session.logits, before))
        self.assertEqual(len(session.history), 1)

    def test_two_ids_cannot_contradict_same_voxel(self):
        session = make_session()
        session.register_point(PointPrompt("positive_same_voxel", (4, 4, 4), "ET", True))
        action = StructuredAction("ET", "include", "r",
                                  positive_point_ids=("positive_same_voxel",),
                                  negative_point_ids=("negative_et",))
        with self.assertRaisesRegex(ValueError, "same voxel"):
            session.apply_action(action)
        self.assertFalse(session.history)

    def test_coordinate_mapping_and_mm_gaussian(self):
        affine = np.array([[0, -2, 0, 100], [1, 0, 0, 30], [0, 0, 3, -20], [0, 0, 0, 1.]])
        world = (affine @ np.array([4, 5, 6, 1]))[:3]
        self.assertEqual(world_to_voxel(world, affine, (10, 10, 10)), (4, 5, 6))
        with self.assertRaises(ValueError):
            world_to_voxel([10000, 1, 1], affine, (10, 10, 10))
        heat = gaussian_from_points((9, 9, 9), [(4, 4, 4)], affine, sigma_mm=2)
        self.assertGreater(float(heat[5, 4, 4]), float(heat[4, 5, 4]))
        self.assertGreater(float(heat[4, 5, 4]), float(heat[4, 4, 5]))

    def test_partial_loss_never_trains_unlabeled_voxels(self):
        logits = torch.zeros((1, 3, 3, 3, 3), requires_grad=True)
        labels = torch.full_like(logits, float("nan"))
        known = torch.zeros_like(logits, dtype=torch.bool)
        known[0, 2, 1, 1, 1] = True
        labels[known] = 0
        loss = sparse_region_bce(logits, labels, known)
        loss.backward()
        self.assertEqual(int((logits.grad != 0).sum()), 1)
        self.assertGreater(float(logits.grad[0, 2, 1, 1, 1]), 0)

    def test_actual_unet_backward_and_checkpoint(self):
        model = TorchPromptCorrector(CorrectorConfig(features=(4, 8), convolutions_per_stage=1))
        mri = torch.randn(4, 8, 8, 8)
        previous = torch.full((3, 8, 8, 8), -1.0)
        gt = torch.zeros_like(previous)
        gt[:, 2:6, 2:6, 2:6] = 1
        example = simulate_training_example(mri, [1, 1, 1, 1], previous, gt, np.eye(4),
                                            target="ET", roi_shape=(8, 8, 8))
        self.assertIsNotNone(example)
        data = example["model_input"].unsqueeze(0)
        self.assertTrue(torch.equal(model(data), torch.zeros_like(previous).unsqueeze(0)))
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        losses = [offline_training_step(model, optimizer, example) for _ in range(3)]
        self.assertTrue(all(np.isfinite(losses)))
        self.assertLess(losses[-1], losses[0])
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.unet.encoder.parameters()))
        no_prompt = data.clone()
        no_prompt[:, 14:] = 0
        self.assertTrue(torch.equal(model(no_prompt), torch.zeros_like(previous).unsqueeze(0)))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "corrector.pt"
            model.eval()
            reference = model(data).detach()
            save_corrector(model, path, {"test": True})
            loaded = load_corrector(path)
            self.assertTrue(torch.equal(loaded(data), reference))
            self.assertTrue(loaded.is_trained_checkpoint)

    def test_feedback_is_sparse_and_test_split_is_rejected(self):
        session = make_session()
        session.source_provenance = {"training_split_verification": {"sha256": "test_hash", "fold": 0}}
        previous = session.logits[:, 2:8, 2:8, 2:8].clone()
        session.apply_action(negative_action())
        with tempfile.TemporaryDirectory() as directory:
            metadata = save_feedback(session, directory, confirmed_by_user=True)
            example, _ = load_feedback(directory)
            self.assertEqual(metadata["source_provenance"], session.source_provenance)
            self.assertEqual(int(example["known"].sum()), 1)
            self.assertTrue(torch.equal(example["previous_logits"], previous))
            self.assertTrue(torch.equal(example["model_input"][8:11], torch.sigmoid(previous)))
            metadata["split"] = "test"
            (Path(directory) / "feedback.json").write_text(json.dumps(metadata))
            with self.assertRaisesRegex(ValueError, "cannot enter training"):
                load_feedback(directory)

    def test_authoritative_split_rejects_self_declared_eval_cases(self):
        with tempfile.TemporaryDirectory() as directory:
            splits = Path(directory) / "splits_final.json"
            splits.write_text(json.dumps([{"train": ["training_case"], "val": ["validation_case"]}]))
            valid = {"case_id": "training_case", "split": "train"}
            metadata = verify_training_split(splits, 0, [valid], [valid])
            self.assertEqual(metadata["fold"], 0)
            self.assertEqual(len(metadata["sha256"]), 64)
            for case_id in ("validation_case", "test_case"):
                falsely_declared = {"case_id": case_id, "split": "train"}
                with self.assertRaisesRegex(ValueError, "not in original fold"):
                    verify_training_split(splits, 0, [falsely_declared], [])
                with self.assertRaisesRegex(ValueError, "not in original fold"):
                    verify_training_split(splits, 0, [], [falsely_declared])

    def test_preflight_rejects_embedded_eval_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.npz"
            np.savez(
                path, mri=np.zeros((4, 2, 2, 2)), modality_mask=np.ones(4),
                previous_logits=np.zeros((3, 2, 2, 2)), affine=np.eye(4),
                atomic_labels=np.zeros((2, 2, 2)), split=np.array("test"),
                case_id=np.array("test_case"),
            )
            with self.assertRaisesRegex(ValueError, "NPZ split conflicts"):
                validate_simulation_identity({"path": str(path), "case_id": "training_case", "split": "train"})

    def test_cli_checks_every_case_before_creating_optimizer(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            splits = directory / "splits_final.json"
            splits.write_text(json.dumps([{"train": ["training_case"], "val": ["validation_case"]}]))
            manifest = directory / "manifest.json"
            manifest.write_text(json.dumps({"simulation": [
                {"case_id": "training_case", "split": "train", "path": "/not_needed_for_membership_check.npz"},
                {"case_id": "validation_case", "split": "train", "path": "/not_needed_for_membership_check.npz"},
            ]}))
            spec = importlib.util.spec_from_file_location(
                "train_corrector_under_test", Path(__file__).with_name("train_corrector.py"))
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            argv = ["train_corrector.py", "--manifest", str(manifest), "--splits-file", str(splits),
                    "--fold", "0", "--output", str(directory / "should_not_exist.pt"), "--steps", "1"]
            with patch.object(sys, "argv", argv), patch.object(torch.optim, "AdamW") as optimizer:
                with self.assertRaisesRegex(ValueError, "not in original fold"):
                    module.main()
                optimizer.assert_not_called()
            self.assertFalse((directory / "should_not_exist.pt").exists())

    def test_box_is_prompt_not_dense_supervision(self):
        session = make_session()
        action = StructuredAction("ET", "exclude", "r", box_prompt=True)
        encoded = session.build_model_input(action)
        self.assertTrue((encoded[:, 20] == -1).all())
        session.apply_action(action)
        self.assertFalse(session.confirmed_mask.any())
        with tempfile.TemporaryDirectory() as directory:
            save_feedback(session, directory, confirmed_by_user=True)
            with self.assertRaisesRegex(ValueError, "boxes alone"):
                load_feedback(directory)

    def test_error_click_uses_fp_and_fn_polarity(self):
        pred = np.zeros((8, 8, 8))
        gt = np.zeros_like(pred)
        gt[2:6, 2:6, 2:6] = 1
        self.assertTrue(simulate_error_click(pred, gt).positive)
        self.assertFalse(simulate_error_click(gt, pred).positive)
        self.assertIsNone(simulate_error_click(gt, gt))


if __name__ == "__main__":
    unittest.main(verbosity=2)
