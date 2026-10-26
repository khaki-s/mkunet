"""Synthetic checks only; never starts training. Run with --mamba on CUDA.

--full-patch verifies the real plan's batch and patch in forward/backward.
JSON contains measurements, not segmentation quality or a convergence claim.
"""
import argparse
import gc
import hashlib
import json
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
from nnunetv2.utilities.brats2021_audit.blocks import (
    raw_to_internal, regions_to_exclusive, ExclusiveHierarchyLoss,
    BoundedModalityGate3D, FrequencySkip3D, ClassConditionalMoE3D)
from nnunetv2.utilities.brats2021_audit.network import build_audit_network
from nnunetv2.utilities.brats2021_audit.sampling import DEFAULT_ROOT, ExclusiveClassDataLoader
from nnunetv2.training.loss.compound_losses import DC_and_BCE_loss
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper


def check_blocks():
    assert raw_to_internal(torch.tensor([0, 1, 2, 4])).tolist() == [0, 2, 1, 3]
    p = torch.rand(2, 3, 4, 5, 6, requires_grad=True)
    q = regions_to_exclusive(p)
    assert (q >= 0).all() and torch.allclose(q.sum(1), torch.ones_like(q[:, 0]))
    (q.square().mean()).backward()
    assert torch.isfinite(p.grad).all()
    x = torch.randn(2, 4, 7, 8, 9)
    gate = BoundedModalityGate3D()
    assert torch.equal(gate(x), x)
    ones = torch.ones(1, 8, 5, 6, 7)
    assert torch.equal(FrequencySkip3D(8)(ones), ones)
    moe = ClassConditionalMoE3D(8)
    feature = torch.randn(1, 8, 4, 5, 6, requires_grad=True)
    out = moe(feature, torch.randn(1, 4, 8, 10, 12))
    out.square().mean().backward()
    assert moe.router[-1].weight.grad.abs().sum() > 0
    assert all(expert[0].weight.grad.abs().sum() > 0 for expert in moe.experts)
    weights = moe.route_sum / moe.route_count
    assert torch.allclose(weights.sum(1), torch.ones(3))
    assert weights[0].argmax() == 3 and weights[1].argmax() == 4 and weights[2].argmax() == 1
    # All-zero image is legal (e.g. padded patch), no all-minus-infinity router.
    assert torch.isfinite(moe(feature, torch.zeros(1, 4, 8, 10, 12))).all()


def region_target(labels):
    return torch.stack((labels > 0, labels >= 2, labels == 3), dim=1).float()


def make_loss(scales):
    base = DC_and_BCE_loss({}, {'batch_dice': False, 'do_bg': True, 'smooth': 1e-5, 'ddp': False},
                           use_ignore_label=False, dice_class=MemoryEfficientSoftDiceLoss)
    weights = np.asarray([.5 ** i for i in range(scales)])
    if scales > 1:
        weights[-1] = 0
    return DeepSupervisionWrapper(ExclusiveHierarchyLoss(base), weights / weights.sum())


def check_losses():
    for labels in (torch.zeros(2, 8, 8, 8, dtype=torch.long),
                   torch.randint(0, 4, (2, 8, 8, 8))):
        # Sample 0 has no NCR, sample 1 has arbitrary classes.
        labels[0][labels[0] == 2] = 1
        targets = [region_target(labels)]
        targets += [F.interpolate(targets[0], size=(4, 4, 4), mode='nearest')]
        logits = [torch.randn_like(y, requires_grad=True) for y in targets]
        loss = make_loss(2)(logits, targets)
        loss.backward()
        assert torch.isfinite(loss) and torch.isfinite(logits[0].grad).all()
    # Hierarchy violation must still get gradients even when clipped difference
    # q_NCR=0: TC low and ET high is not silently treated as an independent class.
    logits = torch.tensor([0., -3., 3.]).view(1, 3, 1, 1, 1).requires_grad_()
    ExclusiveHierarchyLoss(torch.nn.BCEWithLogitsLoss())(logits, torch.zeros_like(logits)).backward()
    assert logits.grad[0, 1].item() != 0 and torch.isfinite(logits.grad).all()


def check_sampler(pm, dataset_json):
    class TinyDataset:
        identifiers = ['train_empty_ncr', 'train_with_ncr']

        def load_case(self, key):
            seg = np.zeros((1, 16, 16, 16), dtype=np.int16)
            seg[0, 3:5, 3:5, 3:5] = 1
            seg[0, 8:10, 8:10, 8:10] = 3
            if key == 'train_with_ncr':
                seg[0, 6:8, 6:8, 6:8] = 2
            coords = {label: np.argwhere(seg == label) for label in (1, 2, 3)}
            return np.zeros((4, 16, 16, 16), np.float32), seg, None, {'class_locations': coords}

    dataset = TinyDataset()
    with tempfile.TemporaryDirectory() as folder:
        cases = {}
        for key in dataset.identifiers:
            coords = dataset.load_case(key)[3]['class_locations']
            np.savez(Path(folder) / f'{key}.npz', **{str(k): v for k, v in coords.items()})
            cases[key] = {'counts': {str(k): len(v) for k, v in coords.items()}}
        manifest = {'train': dataset.identifiers, 'cases': cases}
        loader = ExclusiveClassDataLoader(dataset, 2, (8, 8, 8), (8, 8, 8),
                                          pm.get_label_manager(dataset_json), .33,
                                          index_folder=folder, manifest=manifest)
        counts = np.zeros(4, dtype=int)
        empty_seen = False
        for _ in range(5000):
            keys = loader.get_indices()
            for i, key in enumerate(keys):
                label = loader.draw_labels[i]
                counts[label] += 1
                assert key in dataset.identifiers
                if label == 2:
                    assert key == 'train_with_ncr'
                if label == 0 and key == 'train_empty_ncr':
                    empty_seen = True
        assert np.allclose(counts / counts.sum(), [.5, .125, .25, .125], atol=.02) and empty_seen
        for _ in range(30):
            batch = loader.generate_train_batch()
            for i, label in enumerate(loader.draw_labels):
                if label:
                    assert (batch['target'][i] == label).any()
        return (counts / counts.sum()).tolist()


def run(args):
    torch.set_num_threads(4)
    torch.manual_seed(20260913)
    np.random.seed(20260913)
    root = Path(args.root)
    pm = PlansManager(str(root / 'nnUNetPlans.json'))
    cm = pm.get_configuration('3d_fullres')
    dataset_json = json.loads((root / 'dataset.json').read_text())
    check_blocks()
    check_losses()
    sampler_ratios = check_sampler(pm, dataset_json)
    # Verify class discovery including inference rebuild without initializing a trainer.
    import nnunetv2
    from nnunetv2.utilities.find_class_by_name import recursive_find_python_class
    from nnunetv2.training.nnUNetTrainer.variants.brats2021_audit import nnUNetTrainer_BraTSAudit200 as registry
    trainer_classes = [getattr(registry, name) for name in dir(registry) if name.startswith('nnUNetTrainer_')]
    for cls in trainer_classes:
        found = recursive_find_python_class(str(Path(nnunetv2.__path__[0]) / 'training/nnUNetTrainer'),
                                            cls.__name__, 'nnunetv2.training.nnUNetTrainer')
        assert found is cls
    # Exact common initialization; adapters do not consume RNG before backbone init.
    torch.manual_seed(17)
    baseline = get_network_from_plans(cm.network_arch_class_name, cm.network_arch_init_kwargs,
                                     cm.network_arch_init_kwargs_req_import, 4, 3, deep_supervision=True)
    state = baseline.state_dict()
    for mode in ('baseline', 'gate', 'moe', 'frequency'):
        torch.manual_seed(17)
        model = build_audit_network(cm, 4, 3, True, mode)
        assert all(torch.equal(value, model.state_dict()[key]) for key, value in state.items())
        del model
    del baseline, state
    gc.collect()
    device = torch.device(args.device)
    modes = ['baseline', 'gate', 'moe', 'moe_no_prior', 'moe_uniform', 'frequency']
    if args.mamba:
        modes += ['mamba', 'moe_mamba']
    if args.full_patch:
        modes = [m for m in modes if m not in ('moe_no_prior', 'moe_uniform', 'moe_mamba')]
    shape = tuple(cm.patch_size) if args.full_patch else (32, 32, 32)
    batch = cm.batch_size if args.full_patch else 1
    report = {'torch': torch.__version__, 'device': str(device), 'patch': shape, 'batch': batch,
              'synthetic': True, 'sampler_ratios_bg_ed_ncr_et': sampler_ratios,
              'registered_trainers': [c.__name__ for c in trainer_classes], 'models': {}}
    for mode in modes:
        model = build_audit_network(cm, 4, 3, True, mode).to(device)
        model.train()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        x = torch.randn(batch, 4, *shape, device=device)
        target = region_target(torch.randint(0, 4, (batch, *shape), device=device))
        start = time.perf_counter()
        with torch.autocast(device_type=device.type, enabled=device.type == 'cuda'):
            output = model(x)
            targets = [F.interpolate(target, size=o.shape[2:], mode='nearest') for o in output]
            assert output[0].shape == (batch, 3, *shape)
            loss = make_loss(len(output))(output, targets)
        loss.backward()
        assert torch.isfinite(loss)
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        for name, parameter in model.named_parameters():
            if name.startswith(('context.official', 'moe.experts')):
                assert parameter.grad is not None, name
        if device.type == 'cuda':
            torch.cuda.synchronize()
        stats = {'parameters': sum(p.numel() for p in model.parameters()),
                 'output_shapes': [list(o.shape) for o in output], 'loss': float(loss.detach()),
                 'forward_backward_seconds': time.perf_counter() - start,
                 'peak_allocated_gib': torch.cuda.max_memory_allocated() / 1024**3 if device.type == 'cuda' else None}
        # DS disabled inference + strict state round trip, using deterministic eval.
        model.eval()
        model.decoder.deep_supervision = False
        model.zero_grad(set_to_none=True)
        del output, targets, loss, target, x
        probe = torch.randn(1, 4, 32, 32, 32, device=device)
        with torch.no_grad(), torch.autocast(device_type=device.type, enabled=device.type == 'cuda'):
            expected = model(probe)
        rebuilt = build_audit_network(cm, 4, 3, False, mode).to(device).eval()
        rebuilt.load_state_dict(model.state_dict(), strict=True)
        with torch.no_grad(), torch.autocast(device_type=device.type, enabled=device.type == 'cuda'):
            actual = rebuilt(probe)
        assert torch.equal(actual, expected), mode
        stats['strict_reload_and_ds_off'] = True
        report['models'][mode] = stats
        print(mode, json.dumps(stats), flush=True)
        del model, rebuilt, probe, expected, actual
        gc.collect()
    vendor = Path(__file__).parent / 'vendor'
    meta = json.loads((vendor / 'provenance.json').read_text())
    assert hashlib.sha256((vendor / 'segmambav2.py').read_bytes()).hexdigest() == meta['sha256']
    report['official_source_hash_verified'] = True
    Path(args.output).write_text(json.dumps(report, indent=2) + '\n')
    print('PASS:', args.output, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default=DEFAULT_ROOT)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--mamba', action='store_true')
    parser.add_argument('--full-patch', action='store_true')
    parser.add_argument('--output', default='brats2021_preflight.json')
    run(parser.parse_args())
