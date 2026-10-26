"""CPU inference entry-point checks with synthetic files; no patient data."""
import argparse
import copy
import json
import os
from pathlib import Path
import tempfile
import traceback

_WORK = tempfile.TemporaryDirectory(prefix='brats_feature_inference_')
os.environ['nnUNet_results'] = str(Path(_WORK.name) / 'results')
os.environ.setdefault('nnUNet_raw', '/data02/chujingyi/data/nnUNet_raw')
os.environ.setdefault('nnUNet_preprocessed', '/data02/chujingyi/data/nnUNet_preprocessed')
os.environ['nnUNet_wandb_enabled'] = 'false'
os.environ.setdefault('MPLCONFIGDIR', '/tmp/brats_feature_matplotlib')

import nibabel as nib
import numpy as np
import torch
from nnunetv2.training.nnUNetTrainer.variants.brats2021_audit.nnUNetTrainer_BraTSAgentGeneralistFeature300 import (
    nnUNetTrainer_BraTSAgentGeneralistFeatureLocal300,
    nnUNetTrainer_BraTSAgentGeneralistFeatureComm300,
)
from ...missing_predictor import load_predictor, preprocess_available_case
from ...agent_evaluate import aggregate_case_routes
from .diagnostics import export_feature_attention


def run(plans, dataset):
    records = []
    work = Path(_WORK.name)
    image = np.zeros((28, 32, 24), np.float32)
    image[2:-2, 3:-3, 2:-2] = np.random.default_rng(0).uniform(1, 20, (24, 26, 20))
    affine = np.diag([-1., 1., 1., 1.]); affine[:3, 3] = [23, -17, 8]
    source = work / 'T1ce.nii.gz'
    nib.save(nib.Nifti1Image(image, affine), source)
    for cls in (nnUNetTrainer_BraTSAgentGeneralistFeatureLocal300,
                nnUNetTrainer_BraTSAgentGeneralistFeatureComm300):
        trainer = cls(copy.deepcopy(plans), '3d_fullres', 0, dataset, device=torch.device('cpu'))
        trainer.num_iterations_per_epoch = 1
        trainer.initialize()
        trainer.current_epoch = 100
        trainer.network.set_epoch(100)
        trainer._agent_steps = 101
        trainer.inference_allowed_mirroring_axes = (0,)
        model = Path(trainer.output_folder).parent
        (model / 'plans.json').write_text(json.dumps(trainer.plans_manager.plans))
        (model / 'dataset.json').write_text(json.dumps(dataset))
        trainer.save_checkpoint(str(Path(trainer.output_folder) / 'checkpoint_best.pth'))
        predictor = load_predictor(model, 'checkpoint_best.pth', device=torch.device('cpu'))
        assert predictor.network.decoder.deep_supervision is False
        assert type(predictor.network).__name__ == 'BraTSFeatureInjectedGeneralistUNet'
        case = dict(case_id='synthetic', modalities={'T1ce': str(source)})
        prefix = work / cls.__name__ / 'prediction'
        routes, properties = predictor.predict_available_case(case, prefix)
        output = nib.load(str(prefix) + '.nii.gz')
        assert output.shape == image.shape and np.allclose(output.affine, affine, rtol=0, atol=1e-4)
        rows = aggregate_case_routes(routes, 'synthetic', 2)
        assert len(rows) == 15 and {r['modality'] for r in rows} == {'T1', 'T1ce', 'T2', 'FLAIR', 'Generalist'}
        assert all(r['weight'] == 0. for r in rows if not r['present'])
        assert predictor._case_mask is None
        assert bool(predictor.network.feature_injection_active)
        assert int(predictor.network.stage_epoch) == 100
        tensor, _, mask = preprocess_available_case(case['modalities'], predictor.plans_manager,
                                                    predictor.configuration_manager, dataset)
        first, _ = predictor.predict_with_mask(tensor, mask)
        hidden = tensor.clone(); hidden[~mask] = 10000.
        second, _ = predictor.predict_with_mask(hidden, mask)
        torch.testing.assert_close(first, second, rtol=0, atol=0)
        # Direct planned patch export retains feature axes independently of w.
        patch = torch.randn(1, 4, *trainer.configuration_manager.patch_size)
        with torch.no_grad():
            parts = predictor.network(patch, True, modality_mask=mask[None])
            logits = predictor.network(patch, False, modality_mask=mask[None])
        torch.testing.assert_close(parts['output'], logits, rtol=1e-5, atol=2e-6)
        path = work / cls.__name__ / 'attention.npz'
        export_feature_attention(parts, path, patch_metadata={
            'space': 'synthetic_preprocessed_DHW', 'window_origin': [0, 0, 0], 'mirror_axes_applied': []})
        with np.load(path, allow_pickle=False) as payload:
            assert payload['feature_attention'].shape[:3] == (1, 3, 5)
            assert payload['output_routing'].shape == (1, 3, 5)
            assert json.loads(str(payload['metadata_json']))['axes'][1] == 'latent_attention_head'
        records.append(dict(trainer=cls.__name__, strict_checkpoint_discovery=True,
                            deep_supervision_false=True, single_T1ce_file=True,
                            exported_shape=list(output.shape), affine_preserved=True,
                            tta_forwards=len(routes), routing_rows=len(rows),
                            hidden_channel_invariance=True, aux_and_plain_output_equal=True,
                            feature_attention_npz_export=True))
        print(f"synthetic checkpoint/file/TTA/export {cls.__name__}: passed", flush=True)
        del trainer, predictor, parts
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-root', type=Path,
                        default=Path('/data02/chujingyi/data/nnUNet_preprocessed/Dataset137_BraTS2021'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    plans = json.loads((args.dataset_root / 'nnUNetPlans.json').read_text())
    dataset = json.loads((args.dataset_root / 'dataset.json').read_text())
    plans['continue_training'] = False
    config = plans['configurations']['3d_fullres']
    config['patch_size'], config['batch_size'] = [64, 64, 32], 1
    config['architecture']['arch_kwargs']['features_per_stage'] = [8, 16, 32, 64, 80, 80]
    report = dict(status='running', scope='CPU synthetic file inference; no trained performance claim')
    try:
        report['checks'] = run(plans, dataset)
        report['status'] = 'passed'
    except BaseException:
        report['status'], report['error'] = 'failed', traceback.format_exc()
        raise
    finally:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + '\n')
        _WORK.cleanup()
        print(f"{report['status']}: {args.output}", flush=True)


if __name__ == '__main__':
    main()
