"""Trainer/resume, file inference, evaluation and optional full-plan CUDA checks."""
import argparse
import copy
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import traceback

import nibabel as nib
import numpy as np
import torch
from torch.nn import functional as F

from .agent_evaluate import evaluate_available, file_sha256, paired_comparison, write_json
from .agent_network import build_agent_network
from .convert_semantic_checkpoint import migrate_semantic_weights
from .missing_predictor import MissingModalityPredictor, preprocess_available_case, read_manifest
from .missing_modality import MASK_CODES, mask_from_code, sample_masks
from .reproducibility import seed_experiment, state_digest
from .semantic_network import build_semantic_network
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
from nnunetv2.utilities.find_objects import recursive_find_trainer_class_by_name


def batch_for(trainer, batch_size=2):
    shape = trainer.configuration_manager.patch_size
    x = torch.randn(batch_size, 4, *shape, device=trainer.device)
    label = torch.randint(0, 4, (batch_size, *shape), device=trainer.device)
    regions = torch.stack((label > 0, label > 1, label > 2), 1).float()
    scales = trainer._get_deep_supervision_scales()
    targets = [F.interpolate(regions, size=tuple(int(round(a * b)) for a, b in zip(shape, scale)), mode='nearest') for scale in scales]
    return {'data': x, 'target': targets}


def trainer_checks(plans, dataset_json, device, work):
    from nnunetv2.training.nnUNetTrainer.variants.brats2021_audit import nnUNetTrainer_BraTSModalityAgent300 as registry
    classes = [getattr(registry, name) for name in dir(registry) if name.startswith('nnUNetTrainer_')]
    for cls in classes:
        assert recursive_find_trainer_class_by_name(cls.__name__) is cls
    cls = registry.nnUNetTrainer_BraTSAgentLoMixLoss300
    def make():
        trainer = cls(copy.deepcopy(plans), '3d_fullres', 0, dataset_json, device=device)
        trainer.num_iterations_per_epoch = 1
        trainer.initialize()
        return trainer
    trainer = make()
    assert trainer.num_epochs == 300 and trainer.lr_scheduler.max_steps == 300
    parameter_ids = {id(p) for g in trainer.optimizer.param_groups for p in g['params']}
    assert {id(p) for p in trainer.network.parameters()}.issubset(parameter_ids)
    batch = batch_for(trainer)
    outcomes = []
    for epoch in (49, 99, 119):
        trainer.current_epoch = epoch
        trainer._agent_steps = epoch
        trainer.on_train_epoch_start()
        result = trainer.train_step(batch)
        assert np.isfinite(result['loss'])
        checkpoint = work / f'boundary_{epoch}.pth'
        trainer.save_checkpoint(str(checkpoint))
        restored = make()
        restored.load_checkpoint(checkpoint)
        assert restored.current_epoch == epoch + 1 and restored._agent_steps == epoch + 1
        assert int(restored.network.stage_epoch) == epoch
        assert state_digest(trainer.network) == state_digest(restored.network)
        assert torch.equal(sample_masks(0, trainer._agent_steps, 2), sample_masks(0, restored._agent_steps, 2))
        trainer.current_epoch = restored.current_epoch
        for t in (trainer, restored):
            t.on_train_epoch_start()
            t.train_step(batch)
        assert state_digest(trainer.network) == state_digest(restored.network), f'Resume step diverged at {epoch}'
        outcomes.append({'saved_epoch': epoch, 'next_epoch': epoch + 1, 'next_step_bitwise_equal': True})
        del restored
        gc.collect()
    # The standard nnU-Net --val path must invoke our actual validation override.
    assert cls.perform_actual_validation.__module__.endswith('agent_v1.trainer')
    # Check independent ablations, including baseline optimizer execution.
    for control in (registry.nnUNetTrainer_BraTSModDrop300, registry.nnUNetTrainer_BraTSModDropLoss300,
                    registry.nnUNetTrainer_BraTSSemanticMissing300):
        t = control(copy.deepcopy(plans), '3d_fullres', 0, dataset_json, device=device)
        t.initialize(); t.current_epoch = 50; t.network.set_epoch(50)
        assert np.isfinite(t.train_step(batch)['loss'])
        del t
    seed_experiment(0)
    old = build_semantic_network(trainer.configuration_manager, 4, 3).to(device).eval()
    old.set_epoch(50)
    seed_experiment(1)
    target = build_agent_network(trainer.configuration_manager, communication=True, lomix=True).to(device).eval()
    migration = migrate_semantic_weights(target, {'network_weights': old.state_dict()})
    with torch.no_grad():
        a, b = old(batch['data'][:1])[0], target(batch['data'][:1])[0]
    torch.testing.assert_close(a, b, rtol=1e-5, atol=2e-6)
    del old, target, trainer, batch
    gc.collect()
    return {'registered_classes': [c.__name__ for c in classes], 'all_parameters_in_optimizer': True,
            'boundaries': outcomes, 'moddrop_and_m0_train_steps': True, 'strict_migration': migration}


def inference_checks(plans, dataset_json, device, output):
    manager = PlansManager(plans); config = manager.get_configuration('3d_fullres')
    seed_experiment(0)
    net = build_agent_network(config, communication=True, lomix=True).to(device).eval()
    net.set_epoch(120); net.decoder.deep_supervision = False
    predictor = MissingModalityPredictor(device=device, use_mirroring=True, perform_everything_on_device=False, allow_tqdm=False)
    predictor.manual_initialization(net, manager, config, None, dataset_json, 'synthetic_agent_check', (0,))
    raw = output / 'synthetic_raw'; raw.mkdir(parents=True, exist_ok=True)
    gt = output / 'synthetic_gt'; gt.mkdir(exist_ok=True)
    rng = np.random.default_rng(42)
    shape = (28, 32, 24)
    affine = np.diag([-1., 1., 1., 1.]); affine[:3, 3] = [23, -17, 8]
    modalities = {}
    for m, name in enumerate(('T1', 'T1ce', 'T2', 'FLAIR')):
        image = np.zeros(shape, dtype=np.float32)
        image[2:-2, 3:-3, 2:-2] = rng.uniform(1, 20, (24, 26, 20)) + m
        path = raw / (name + '.nii.gz'); nib.save(nib.Nifti1Image(image, affine), path); modalities[name] = str(path)
    label = np.zeros(shape, np.uint8); label[5:20, 6:25, 5:20] = 1; label[8:18, 9:20, 8:17] = 2; label[10:15, 12:17, 10:15] = 3
    nib.save(nib.Nifti1Image(label, affine), gt / 'synthetic.nii.gz')
    manifest = output / 'single_file_manifest.json'
    write_json(manifest, {'case_id': 'single_T1ce', 'modalities': {'T1ce': modalities['T1ce']}})
    case = read_manifest(manifest)[0]
    data, props, mask = preprocess_available_case(case['modalities'], manager, config, dataset_json)
    assert mask.tolist() == [False, True, False, False]
    assert torch.count_nonzero(data[[0, 2, 3]]) == 0
    prefix = output / 'single_file' / case['case_id']
    routes, _ = predictor.predict_available_case(case, prefix)
    pred = nib.load(str(prefix) + '.nii.gz')
    assert pred.shape == shape and np.allclose(pred.affine, affine, rtol=0, atol=1e-4)
    assert predictor._case_mask is None
    with torch.no_grad():
        a, _ = predictor.predict_with_mask(data, mask)
        hidden = data.clone(); hidden[~mask] = 1e3
        b, _ = predictor.predict_with_mask(hidden, mask)
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    # Intentional invalid geometry is rejected before model execution.
    bad = raw / 'bad.nii.gz'; bad_affine = affine.copy(); bad_affine[0, 3] += 10
    nib.save(nib.Nifti1Image(np.ones(shape, np.float32), bad_affine), bad)
    try:
        preprocess_available_case({'T1': modalities['T1'], 'T2': str(bad)}, manager, config, dataset_json)
    except ValueError:
        pass
    else:
        raise AssertionError('Mismatched geometry accepted')
    evaluation = output / 'synthetic_evaluation_15m'
    summary = evaluate_available(predictor, [{'case_id': 'synthetic', 'modalities': modalities}], gt, evaluation,
                                 provenance={'fixture': 'synthetic; not research performance'})
    assert summary['n_masks'] == 15
    paired = paired_comparison(evaluation / 'per_case_metrics.csv', evaluation / 'per_case_metrics.csv', evaluation / 'paired_comparison.json', repetitions=30)
    assert all(v['macro_delta'] == 0 for v in paired['regions'].values())
    script = Path(__file__).resolve().parents[3] / 'scripts/brats2021_audit/plot_agent_contributions.py'
    subprocess.run([sys.executable, str(script), '--evaluation', str(evaluation)], check=True)
    for filename in ('routing_heatmap.pdf', 'routing_heatmap.png', 'beta_heatmap.pdf', 'contribution_delta_heatmap.pdf'):
        assert (evaluation / filename).stat().st_size > 1000
    del net, predictor
    gc.collect()
    return {'single_file_normalization_zero_slots': True, 'geometry_export_and_tta': True,
            'hidden_slot_invariance': True, 'geometry_mismatch_rejected': True,
            'all15_full_volume_evaluation': True, 'paired_self_comparison_zero': True,
            'plots_generated_from_csv': True, 'synthetic_evaluation': str(evaluation), 'tta_forwards_single_case': len(routes)}


def full_patch_check(plans, dataset_json, device, output, real_image=None, legacy_checkpoint=None):
    if device.type != 'cuda':
        raise ValueError('Full-plan check requires CUDA')
    from nnunetv2.training.nnUNetTrainer.variants.brats2021_audit.nnUNetTrainer_BraTSModalityAgent300 import nnUNetTrainer_BraTSAgentLoMixLoss300
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    trainer = nnUNetTrainer_BraTSAgentLoMixLoss300(copy.deepcopy(plans), '3d_fullres', 0, dataset_json, device=device)
    trainer.initialize(); trainer.current_epoch = 120; trainer.network.set_epoch(120)
    batch = batch_for(trainer, 2)
    started = time.monotonic()
    trainer.optimizer.zero_grad(set_to_none=True)
    with torch.autocast('cuda'):
        parts = trainer.network(batch['data'], True, modality_mask=torch.ones(2, 4, device=device))
        loss, components = trainer.compute_agent_loss(parts, batch['target'])
    trainer.grad_scaler.scale(loss).backward(); trainer.grad_scaler.unscale_(trainer.optimizer)
    assert torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in trainer.network.parameters() if p.grad is not None)
    assert trainer.network.communication.rho.grad.abs().max() > 0
    assert trainer.network.lomix.alpha[0].grad.abs().max() > 0
    torch.nn.utils.clip_grad_norm_(trainer.network.parameters(), 12)
    trainer.grad_scaler.step(trainer.optimizer); trainer.grad_scaler.update()
    torch.cuda.synchronize()
    report = {'patch': trainer.configuration_manager.patch_size, 'batch_size': 2, 'modalities': 'all four',
              'amp': True, 'deterministic': torch.are_deterministic_algorithms_enabled(),
              'losses': components, 'peak_allocated_GiB': torch.cuda.max_memory_allocated() / 2**30,
              'peak_reserved_GiB': torch.cuda.max_memory_reserved() / 2**30, 'seconds': time.monotonic() - started}
    del parts, loss, batch
    trainer.optimizer.zero_grad(set_to_none=True)
    trainer.network.eval(); trainer.network.decoder.deep_supervision = False
    if legacy_checkpoint:
        saved = torch.load(legacy_checkpoint, map_location='cpu', weights_only=False)
        old = build_semantic_network(trainer.configuration_manager, 4, 3, False).to(device).eval()
        old.load_state_dict(saved['network_weights'], strict=True)
        image = torch.randn(1, 4, *trainer.configuration_manager.patch_size, device=device)
        with torch.no_grad(), torch.autocast('cuda'):
            reference = old(image).cpu()
        del old
        gc.collect(); torch.cuda.empty_cache()
        migrate_semantic_weights(trainer.network, saved)
        with torch.no_grad(), torch.autocast('cuda'):
            converted = trainer.network(image).cpu()
        torch.testing.assert_close(converted, reference, atol=1e-5, rtol=1e-5)
        report['real_checkpoint_migration'] = {'source_sha256': file_sha256(legacy_checkpoint),
                                             'max_abs_output_difference': float((converted - reference).abs().max())}
        del saved, image, reference, converted
    if real_image:
        predictor = MissingModalityPredictor(device=device, use_mirroring=True, perform_everything_on_device=False, allow_tqdm=False)
        predictor.manual_initialization(trainer.network, trainer.plans_manager, trainer.configuration_manager, None,
                                         dataset_json, type(trainer).__name__, (0,))
        case = {'case_id': 'real_single_T2_check', 'modalities': {'T2': str(real_image)}}
        prefix = output / 'real_single_file' / case['case_id']
        routes, props = predictor.predict_available_case(case, prefix)
        source, pred = nib.load(str(real_image)), nib.load(str(prefix) + '.nii.gz')
        assert source.shape == pred.shape and np.allclose(source.affine, pred.affine, rtol=0, atol=1e-4)
        report['real_single_file'] = {'input': str(real_image), 'output': str(prefix) + '.nii.gz',
            'geometry_preserved': True, 'modality_mask': props['modality_mask'], 'tta_axes': [0],
            'n_window_tta_forwards': len(routes), 'purpose': 'technical smoke check, not missing-modality-trained result'}
    del trainer
    gc.collect(); torch.cuda.empty_cache()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-root', type=Path, default=Path('/data02/chujingyi/data/nnUNet_preprocessed/Dataset137_BraTS2021'))
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--full-patch', action='store_true')
    parser.add_argument('--full-only', action='store_true')
    parser.add_argument('--real-image', type=Path)
    parser.add_argument('--legacy-checkpoint', type=Path)
    args = parser.parse_args()
    torch.set_num_threads(4)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {'status': 'running', 'device': args.device, 'torch': torch.__version__}
    write_json(args.output_dir / 'report.json', report)
    plans = json.loads((args.dataset_root / 'nnUNetPlans.json').read_text())
    plans['continue_training'] = False  # Local CLI normally adds this trainer-only flag.
    dataset = json.loads((args.dataset_root / 'dataset.json').read_text())
    small = copy.deepcopy(plans); c = small['configurations']['3d_fullres']
    c['patch_size'], c['batch_size'] = [64, 64, 32], 2
    c['architecture']['arch_kwargs']['features_per_stage'] = [8, 16, 32, 64, 80, 80]
    try:
        with tempfile.TemporaryDirectory(prefix='brats_agent_checks_') as work:
            # The CLI environment already routes trainer checkpoints into a temporary result root.
            if not args.full_only:
                report['trainer'] = trainer_checks(small, dataset, torch.device(args.device), Path(work))
                write_json(args.output_dir / 'report.json', report)
                report['inference'] = inference_checks(small, dataset, torch.device(args.device), args.output_dir)
                write_json(args.output_dir / 'report.json', report)
            if args.full_patch or args.full_only:
                report['full_patch'] = full_patch_check(plans, dataset, torch.device(args.device), args.output_dir,
                                                        args.real_image, args.legacy_checkpoint)
            report['status'] = 'passed'
    except BaseException:
        report['status'], report['error'] = 'failed', traceback.format_exc()
        raise
    finally:
        write_json(args.output_dir / 'report.json', report)
        print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
