"""Small CPU optimizer test for explicit ET-only training, without patient data."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from nnunetv2.utilities.brats2021_audit_v2.interactive.model import load_corrector


class TrainingTargetTests(unittest.TestCase):
    def test_et_only_and_explicit_bound_survive_checkpoint(self):
        torch.set_num_threads(1)
        specification = importlib.util.spec_from_file_location(
            "train_corrector_target_test", Path(__file__).with_name("train_corrector.py"))
        module = importlib.util.module_from_spec(specification)
        specification.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            splits = directory / "splits.json"
            splits.write_text(json.dumps([{"train": ["synthetic_train"], "val": ["validation"]}]))
            example = directory / "case.npz"
            labels = np.zeros((8, 8, 8), np.uint8)
            labels[2:6, 2:6, 2:6] = 3
            np.savez(example, mri=np.ones((4, 8, 8, 8), np.float32),
                     modality_mask=np.ones(4, bool), affine=np.eye(4),
                     previous_logits=np.full((3, 8, 8, 8), -1., np.float32),
                     atomic_labels=labels, case_id=np.array("synthetic_train"), split=np.array("train"))
            manifest = directory / "manifest.json"
            manifest.write_text(json.dumps({"simulation": [{
                "case_id": "synthetic_train", "split": "train", "path": str(example)}]}))
            output = directory / "trained.pth"
            argv = ["train_corrector.py", "--manifest", str(manifest), "--splits-file", str(splits),
                    "--fold", "0", "--output", str(output), "--steps", "2", "--rounds-per-case", "1",
                    "--features", "4", "8", "--roi-size", "8", "8", "8",
                    "--targets", "ET", "--max-delta", "16"]
            with patch.object(sys, "argv", argv), contextlib.redirect_stdout(io.StringIO()) as stream:
                with patch.object(module, "simulate_error_click", wraps=module.simulate_error_click) as sampler:
                    module.main()
            self.assertTrue(sampler.call_args_list)
            self.assertTrue(all(call.args[2] == "ET" for call in sampler.call_args_list))
            records = [json.loads(line) for line in stream.getvalue().splitlines() if line.startswith("{")]
            self.assertEqual([row["target"] for row in records], ["ET", "ET"])
            self.assertTrue(all(np.isfinite(row["loss"]) for row in records))
            payload = torch.load(output, map_location="cpu", weights_only=True)
            self.assertEqual(payload["training_metadata"]["simulation_targets"], ["ET"])
            model = load_corrector(output)
            self.assertTrue(model.is_trained_checkpoint)
            self.assertEqual(model.config.max_delta, 16.)
            # Resume must preserve the checkpoint's contract, not silently change it.
            with patch.object(sys, "argv", argv + ["--initialize-from", str(output)]):
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as raised:
                    module.main()
                self.assertEqual(raised.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
