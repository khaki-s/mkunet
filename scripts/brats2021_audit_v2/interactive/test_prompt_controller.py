"""Explicit box/text controller regression checks; no real LLM or patient data."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from nnunetv2.utilities.brats2021_audit_v2.interactive.backends import NNInteractiveBackend
from nnunetv2.utilities.brats2021_audit_v2.interactive.controller import LLMController, RuleController
from nnunetv2.utilities.brats2021_audit_v2.interactive.schemas import BoxPrompt, PointPrompt, SpatialROI
from nnunetv2.utilities.brats2021_audit_v2.interactive.session import CorrectionSession


class ZeroCorrector(torch.nn.Module):
    def forward(self, image):
        return image.new_zeros((image.shape[0], 3, *image.shape[2:]))


class StubGenerator:
    def __init__(self, action):
        self.action = action
        self.observation = None

    def generate(self, system, user, *, images=()):
        self.observation = json.loads(user)
        return json.dumps(self.action)


def make_session():
    session = CorrectionSession(
        np.ones((4, 10, 10, 10), np.float32), [1, 1, 1, 1],
        np.ones((3, 10, 10, 10), np.float32), np.eye(4), ZeroCorrector())
    session.register_roi("selected", SpatialROI((2, 2, 2), (8, 8, 8)))
    session.register_roi("other", SpatialROI((1, 1, 1), (9, 9, 9)))
    session.register_point(PointPrompt("negative", (4, 4, 4), "ET", False))
    session.register_box(BoxPrompt("box_et", (3, 3, 4), (7, 7, 5), "ET", False))
    session.register_box(BoxPrompt("box_other", (3, 3, 5), (7, 7, 6), "ET", False))
    return session


class ExplicitBoxControllerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_rule_box_only_is_not_dense_supervision(self):
        for text, operation, sign in (("框内ET漏分了", "include", 1),
                                      ("框内ET分多了", "exclude", -1)):
            session = make_session()
            action = RuleController().propose(text, session, roi_id="selected", box_prompt=True)
            self.assertEqual(action.operation, operation)
            self.assertTrue(action.box_prompt)
            self.assertTrue((session.build_model_input(action)[:, 20] == sign).all())
            session.apply_action(action)
            self.assertFalse(session.confirmed_mask.any(), "A box must not create voxel labels")

    def test_roi_and_word_box_do_not_create_annotation(self):
        session = make_session()
        action = RuleController().propose("框内ET分多了", session, roi_id="selected",
                                          point_ids=("negative",))
        self.assertFalse(action.box_prompt)
        self.assertFalse(session.build_model_input(action)[:, 20].any())
        with self.assertRaises(ValueError):
            RuleController().propose("框内ET分多了", session, roi_id="selected")

    def test_llm_preserves_selected_real_box(self):
        session = make_session()
        expected = RuleController().propose("框内ET分多了", session, roi_id="selected",
                                            box_prompt=True, box_id="box_et")
        generator = StubGenerator(expected.to_dict())
        actual = LLMController(generator).propose(
            "框内ET分多了", session, roi_id="selected", target="ET",
            box_prompt=True, box_id="box_et")
        self.assertEqual(actual, expected)
        self.assertTrue(generator.observation["box_prompt"])
        self.assertEqual(generator.observation["selected_box_id"], "box_et")
        self.assertEqual(generator.observation["selected_points"], {})
        self.assertFalse(session.history)
        model_input = session.build_model_input(actual)
        self.assertEqual(tuple(model_input.shape), (1, 21, 6, 6, 6))
        encoded = model_input[0, 20]
        self.assertEqual(int((encoded == -1).sum()), 16)
        self.assertTrue((encoded[1:5, 1:5, 2:3] == -1).all())
        session.apply_action(actual)
        self.assertFalse(session.confirmed_mask.any())

    def test_llm_cannot_invent_remove_or_move_box(self):
        session = make_session()
        base = RuleController().propose("ET分多了", session, roi_id="selected",
                                       point_ids=("negative",)).to_dict()
        for supplied, returned in ((False, True), (True, False)):
            generator = StubGenerator({**base, "box_prompt": returned})
            with self.assertRaisesRegex(ValueError, "create or remove"):
                LLMController(generator).propose("ET分多了", session, roi_id="selected",
                    point_ids=("negative",), target="ET", box_prompt=supplied)
        moved = StubGenerator({**base, "box_prompt": True, "roi_id": "other"})
        with self.assertRaisesRegex(ValueError, "selected ROI"):
            LLMController(moved).propose("ET分多了", session, roi_id="selected",
                point_ids=("negative",), target="ET", box_prompt=True)
        switched = StubGenerator({**base, "box_prompt": True, "box_id": "box_other"})
        with self.assertRaisesRegex(ValueError, "selected box"):
            LLMController(switched).propose("ET分多了", session, roi_id="selected",
                point_ids=("negative",), target="ET", box_prompt=True, box_id="box_et")
        self.assertFalse(session.history)

    def test_bad_flag_and_box_replacement_rejected(self):
        session = make_session()
        for flag in (1, "true", None):
            for controller in (RuleController(), LLMController(StubGenerator({}))):
                with self.assertRaisesRegex(ValueError, "boolean"):
                    controller.propose("ET分多了", session, roi_id="selected", box_prompt=flag)
        with self.assertRaisesRegex(ValueError, "not dense ground truth"):
            RuleController().propose("ET改成NCR", session, roi_id="selected",
                                     point_ids=("negative",), box_prompt=True)
        point_replacement = RuleController().propose(
            "ET改成NCR", session, roi_id="selected", point_ids=("negative",))
        self.assertEqual(point_replacement.replacement_class, "NCR")

    def test_registered_box_target_polarity_and_roi_are_binding(self):
        session = make_session()
        with self.assertRaisesRegex(ValueError, "target/polarity"):
            RuleController().propose("ET漏分了", session, roi_id="selected",
                                     box_prompt=True, box_id="box_et")
        with self.assertRaisesRegex(ValueError, "target/polarity"):
            RuleController().propose("TC分多了", session, roi_id="selected",
                                     box_prompt=True, box_id="box_et")
        session.register_roi("tiny", SpatialROI((4, 4, 4), (6, 6, 6)))
        with self.assertRaisesRegex(ValueError, "inside"):
            RuleController().propose("ET分多了", session, roi_id="tiny",
                                     box_prompt=True, box_id="box_et")
        for box_id in ("unknown", ["box_et"]):
            with self.assertRaises(ValueError):
                LLMController(StubGenerator({})).propose(
                    "ET分多了", session, roi_id="selected", box_prompt=True, box_id=box_id)
        self.assertFalse(session.history)

    def test_official_backend_receives_explicit_negative_box(self):
        class StubEngine:
            def set_image(self, image):
                self.shape = image.shape[1:]

            def set_target_buffer(self, output):
                self.output = output

            def add_initial_seg_interaction(self, initial, run_prediction):
                self.output.copy_(torch.from_numpy(initial))

            def add_bbox_interaction(self, bounds, include_interaction, run_prediction):
                self.received = (bounds, include_interaction, run_prediction)

        session = make_session()
        engine = StubEngine()
        session.corrector = NNInteractiveBackend(
            None, session.mri[1:2].numpy(), session.affine,
            modality_index=1, inference_session=engine)
        action = RuleController().propose("框内ET分多了", session, roi_id="selected",
                                          box_prompt=True, box_id="box_et")
        session.apply_action(action)
        self.assertEqual(engine.received, ([[1, 5], [1, 5], [2, 3]], False, True))
        self.assertFalse(session.confirmed_mask.any())

    def test_official_backend_rejects_legacy_roi_box(self):
        session = make_session()
        # The rejection must happen before invoking any external library method.
        session.corrector = NNInteractiveBackend(
            None, session.mri[1:2].numpy(), session.affine,
            modality_index=1, inference_session=object())
        action = RuleController().propose("框内ET分多了", session, roi_id="selected", box_prompt=True)
        with self.assertRaises(ValueError):
            session.apply_action(action)
        self.assertFalse(session.history)

    def test_official_backend_rejects_3d_box_without_capability(self):
        session = make_session()
        session.register_box(BoxPrompt("volume_box", (3, 3, 3), (7, 7, 7), "ET", False))
        session.corrector = NNInteractiveBackend(
            None, session.mri[1:2].numpy(), session.affine,
            modality_index=1, inference_session=object())
        action = RuleController().propose("ET分多了", session, roi_id="selected",
                                          box_prompt=True, box_id="volume_box")
        with self.assertRaisesRegex(ValueError, "2D slice boxes"):
            session.apply_action(action)
        self.assertFalse(session.history)

    def test_text_box_cli_and_replay_keep_flag(self):
        path = Path(__file__).with_name("correct_case.py")
        spec = importlib.util.spec_from_file_location("correct_case_box_test", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            case = root / "case.npz"
            session = make_session()
            np.savez_compressed(case, mri=session.mri.numpy(), logits=session.logits.numpy(),
                affine=session.affine, modality_mask=session.modality_mask.numpy(),
                case_id=np.array("synthetic"), split=np.array("validation"))
            prompts = root / "prompts.json"
            prompts.write_text(json.dumps({
                "rois": {"selected": {"start": [2, 2, 2], "stop": [8, 8, 8]}},
                "boxes": [{"box_id": "box_et", "start": [3, 3, 4], "stop": [7, 7, 5],
                           "target": "ET", "positive": False}],
                "commands": [{"text": "ET分多了", "target": "ET", "roi_id": "selected",
                              "box_prompt": True, "box_id": "box_et"}],
            }))
            output = root / "result"
            argv = [str(path), "--case", str(case), "--interactions", str(prompts),
                    "--output", str(output), "--demo-untrained"]
            with patch.object(sys, "argv", argv):
                module.main()
            replay = json.loads((output / "replay.json").read_text())
            self.assertTrue(replay["commands"][0]["action"]["box_prompt"])
            self.assertEqual(replay["commands"][0]["action"]["box_id"], "box_et")
            self.assertEqual(replay["boxes"][0]["box_id"], "box_et")
            report = json.loads((output / "session.json").read_text())
            self.assertEqual(report["operations"][0]["confirmed_region_voxels"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
