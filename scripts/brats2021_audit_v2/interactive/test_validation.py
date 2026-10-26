"""CPU integration checks with synthetic volumes; no patient data or training."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import nibabel as nib
import numpy as np
import torch

from nnunetv2.utilities.brats2021_audit_v2.interactive.model import (
    CorrectorConfig, TorchPromptCorrector, save_corrector,
)


class ValidationWorkflowTest(unittest.TestCase):
    def test_fixed_prompts_no_errors_and_split_guard(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prepared, gt_dir = root / "prepared", root / "gt"
            prepared.mkdir()
            gt_dir.mkdir()
            ids = ["case_error", "case_perfect"]
            gt = np.zeros((12, 12, 12), np.uint8)
            gt[2:10, 2:10, 2:10] = 1
            gt[4:8, 4:8, 4:8] = 3
            initial = np.stack((gt > 0, gt >= 2, gt == 3)).astype(np.float32) * 4 - 2
            for case in ids:
                logits = initial.copy()
                if case == "case_error":
                    logits[2] = -2
                    logits[2, 0:2, 0:2, 0:2] = 2
                np.savez_compressed(
                    prepared / (case + ".npz"), mri=np.ones((4, 12, 12, 12), np.float32),
                    raw_mri=np.ones((4, 12, 12, 12), np.float32),
                    logits=logits, modality_mask=np.ones(4, bool), affine=np.eye(4),
                    case_id=np.array(case), split=np.array("validation"))
                nib.save(nib.Nifti1Image(gt, np.eye(4)), gt_dir / (case + ".nii.gz"))
            splits = root / "splits.json"
            splits.write_text(json.dumps([{"train": ["train_case"], "val": ids}]))
            manifest = root / "cases.json"
            manifest.write_text(json.dumps({"cases": [{"case_id": case} for case in ids]}))
            checkpoint = root / "synthetic_identity_corrector.pth"
            model = TorchPromptCorrector(CorrectorConfig(features=(2, 4)))
            # Fixture deliberately exercises inference + constraints, not quality.
            model.is_trained_checkpoint = True
            save_corrector(model, checkpoint, {"synthetic_test_fixture": True})
            digest_before = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            command = [
                sys.executable, str(Path(__file__).with_name("validate_corrector.py")),
                "--manifest", str(manifest), "--prepared", str(prepared), "--gt", str(gt_dir),
                "--splits-file", str(splits), "--corrector-checkpoint", str(checkpoint),
                "--click-budgets", "1", "3", "--roi-size", "10", "10", "10", "--device", "cpu",
            ]
            oracle, replay = root / "oracle", root / "replay"
            subprocess.run(command + ["--output", str(oracle)], check=True, capture_output=True, text=True)
            subprocess.run(command + ["--output", str(replay), "--replay-dir", str(oracle)],
                           check=True, capture_output=True, text=True)
            self.assertEqual((oracle / "per_case_metrics.csv").read_text(),
                             (replay / "per_case_metrics.csv").read_text())
            self.assertEqual(digest_before, hashlib.sha256(checkpoint.read_bytes()).hexdigest())
            perfect = json.loads((oracle / "cases/case_perfect/prompt_sequence.json").read_text())
            self.assertEqual(perfect["actual_clicks"], 0)
            self.assertEqual(perfect["commands"], [])
            sequence = json.loads((oracle / "cases/case_error/prompt_sequence.json").read_text())
            self.assertEqual(sequence["actual_clicks"], 3)
            self.assertEqual(len(sequence["commands"]), 3)
            self.assertTrue(all("text" in row and "action" not in row for row in sequence["commands"]))
            self.assertTrue(all(p["target"] == "ET" for p in sequence["points"]))
            for case in ids:
                for budget in (0, 1, 3):
                    a = nib.load(oracle / f"cases/{case}/clicks_{budget:02d}.nii.gz").get_fdata()
                    b = nib.load(replay / f"cases/{case}/clicks_{budget:02d}.nii.gz").get_fdata()
                    np.testing.assert_array_equal(a, b)
            self.assertEqual(json.loads((oracle / "manifest.json").read_text())["status"], "complete")
            # A train case must be rejected before an output directory is created.
            manifest.write_text(json.dumps({"cases": [{"case_id": "train_case"}]}))
            invalid = root / "invalid"
            rejected = subprocess.run(command + ["--output", str(invalid)], capture_output=True, text=True)
            self.assertNotEqual(rejected.returncode, 0)
            self.assertIn("VALIDATION", rejected.stderr)
            self.assertFalse(invalid.exists())

            # Final test evaluation must explicitly select a disjoint set and
            # reuse the completed validation protocol, without fitting on it.
            test_id = "case_test"
            with np.load(prepared / "case_error.npz") as source:
                data = {key: source[key].copy() for key in source.files}
            data.update(case_id=np.array(test_id), split=np.array("test"))
            np.savez_compressed(prepared / (test_id + ".npz"), **data)
            nib.save(nib.Nifti1Image(gt, np.eye(4)), gt_dir / (test_id + ".nii.gz"))
            test_manifest = root / "test_cases.json"
            test_manifest.write_text(json.dumps({"cases": [{"case_id": test_id}]}))
            test_command = list(command)
            test_command[test_command.index("--manifest") + 1] = str(test_manifest)
            test_command.extend(["--split", "test", "--test-manifest", str(test_manifest),
                                 "--protocol-from", str(oracle / "manifest.json")])
            subprocess.run(test_command + ["--output", str(root / "test_result")],
                           check=True, capture_output=True, text=True)
            result = json.loads((root / "test_result/manifest.json").read_text())
            self.assertEqual(result["split"], "test")
            self.assertEqual(result["status"], "complete")
            changed = subprocess.run(
                test_command + ["--roi-size", "8", "8", "8", "--output", str(root / "changed_protocol")],
                capture_output=True, text=True)
            self.assertNotEqual(changed.returncode, 0)
            self.assertIn("Frozen validation settings differ", changed.stderr)
            self.assertFalse((root / "changed_protocol").exists())

            # Exercise the official-backend dispatch without installing its
            # package/downloading weights: the dummy tests adapter plumbing only.
            manifest.write_text(json.dumps({"cases": [{"case_id": case} for case in ids]}))
            official_folder = root / "dummy_official"
            official_folder.mkdir()
            (official_folder / "checkpoint_final.pth").write_bytes(b"synthetic fixture")
            spec = importlib.util.spec_from_file_location(
                "validation_entry", Path(__file__).with_name("validate_corrector.py"))
            entry = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(entry)
            engines = []

            class DummyOfficialBackend:
                def __init__(self, folder, raw, affine, *, modality_index, device, inference_session):
                    self.engine = inference_session if inference_session is not None else object()
                    engines.append((self.engine, inference_session is None))
                    assert raw.shape == (1, 12, 12, 12) and modality_index == 1

                def predict_delta(self, model_input, **kwargs):
                    return torch.zeros((1, 3, *model_input.shape[2:]))

            official_command = command[1:]
            offset = official_command.index("--corrector-checkpoint")
            del official_command[offset:offset + 2]
            official_command.extend(["--nninteractive-model", str(official_folder),
                                     "--nninteractive-modality", "1",
                                     "--output", str(root / "official_dispatch")])
            from nnunetv2.utilities.brats2021_audit_v2.interactive import backends
            with patch.object(backends, "NNInteractiveBackend", DummyOfficialBackend), \
                    patch.object(sys, "argv", official_command):
                entry.main()
            self.assertEqual(len(engines), 2)
            self.assertIs(engines[0][0], engines[1][0])
            self.assertEqual([first for _, first in engines], [True, False])
            self.assertEqual((root / "official_dispatch/per_case_metrics.csv").read_text(),
                             (oracle / "per_case_metrics.csv").read_text())


if __name__ == "__main__":
    unittest.main()
