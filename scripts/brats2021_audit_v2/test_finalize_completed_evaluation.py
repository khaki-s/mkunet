import json
from pathlib import Path
import tempfile
import unittest

from nnunetv2.utilities.brats2021_audit_v2.agent_evaluate import file_sha256, REGIONS
from scripts.brats2021_audit_v2.finalize_completed_evaluation import finalize


class FinalizeTests(unittest.TestCase):
    def fixture(self, root):
        weight, gt = root / 'weight.pth', root / 'case.nii.gz'
        weight.write_bytes(b'weight')
        gt.write_bytes(b'gt')
        pred = root / 'predictions/mask_15/case.nii.gz'
        pred.parent.mkdir(parents=True)
        pred.write_bytes(b'prediction')
        done = root / 'completed/mask_15/case.json'
        done.parent.mkdir(parents=True)
        rows = [dict(case_id='case', mask_code=15, region=name, dice=1.,
                     ref_mm3=1., fp_mm3=0., hd95_mm=0., hd95_status='finite') for name in REGIONS]
        done.write_text(json.dumps(dict(metrics=rows, routing=[], prediction_sha256=file_sha256(pred))))
        contract = dict(case_ids=['case'], mask_codes=[15], gt_folder=str(root),
                        gt_sha256={'case':file_sha256(gt)},
                        provenance=dict(checkpoint=str(weight),checkpoint_sha256=file_sha256(weight)))
        (root/'manifest.json').write_text(json.dumps(dict(status='running',contract=contract)))
        return contract, pred

    def test_mask15_only_can_finalize_empty_removal_comparison(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            contract,_=self.fixture(root)
            result=finalize(root)
            self.assertEqual(result['n_cases'],1)
            self.assertEqual((root/'input_removal_delta.csv').read_text().count('\n'),1)
            saved=json.loads((root/'manifest.json').read_text())
            self.assertEqual(saved,dict(status='complete',contract=contract))
            self.assertTrue((root/'manifest.before_summary_finalization.json').exists())

    def test_changed_prediction_never_finalizes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            _,pred=self.fixture(root)
            pred.write_bytes(b'tampered')
            with self.assertRaisesRegex(ValueError,'Prediction changed'):
                finalize(root)
            self.assertEqual(json.loads((root/'manifest.json').read_text())['status'],'running')
            self.assertFalse((root/'summary.json').exists())


if __name__ == '__main__':
    unittest.main()
