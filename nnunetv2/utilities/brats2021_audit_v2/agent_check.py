"""Bounded synthetic checks for modality agents; never starts formal training."""
import argparse
import copy
import gc
import json
import os
from pathlib import Path
import tempfile
import traceback

import numpy as np
import torch
from torch.nn import functional as F

from .missing_modality import MASK_CODES, mask_from_code, mask_codes, sample_masks, apply_modality_mask
from .agent_network import build_agent_network
from .agent_losses import positive_region_loss, present_expert_loss
from .agent_resize import _DeterministicTrilinear
from .semantic_network import build_semantic_network
from .reproducibility import seed_experiment


def run_core(root, device):
    from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
    plans = json.loads((root / 'nnUNetPlans.json').read_text())
    small = copy.deepcopy(plans)
    cfg = small['configurations']['3d_fullres']
    cfg['patch_size'], cfg['batch_size'] = [64, 64, 32], 2
    cfg['architecture']['arch_kwargs']['features_per_stage'] = [8, 16, 32, 64, 80, 80]
    config = PlansManager(small).get_configuration('3d_fullres')
    checks = {}
    masks = torch.stack([mask_from_code(c) for c in MASK_CODES])
    assert mask_codes(masks).tolist() == list(MASK_CODES)
    assert masks.sum(1).bincount().tolist() == [0, 4, 6, 4, 1]
    try:
        apply_modality_mask(torch.zeros(1, 4, 2, 2, 2), torch.zeros(1, 4))
    except ValueError:
        pass
    else:
        raise AssertionError('Empty availability was accepted')
    seed_experiment(0)
    state = torch.get_rng_state().clone()
    a = sample_masks(12, 99, 128)
    assert torch.equal(a, sample_masks(12, 99, 128)) and torch.equal(state, torch.get_rng_state())
    assert not torch.equal(a, sample_masks(12, 100, 128))
    checks['mask_encoding_sampling_and_rng_isolation'] = True
    x = torch.randn(1, 2, 3, 5, 4, dtype=torch.float64, requires_grad=True)
    y = x.detach().clone().requires_grad_()
    g = torch.randn(1, 2, 7, 6, 9, dtype=torch.float64)
    _DeterministicTrilinear.apply(x, (7, 6, 9)).backward(g)
    F.interpolate(y, size=(7, 6, 9), mode='trilinear', align_corners=False).backward(g)
    torch.testing.assert_close(x.grad, y.grad, rtol=1e-12, atol=1e-12)
    checks['deterministic_resize_adjoint_matches_native'] = True
    seed_experiment(0)
    old = build_semantic_network(config, 4, 3).to(device).eval()
    old.set_epoch(50)
    seed_experiment(0)
    net = build_agent_network(config).to(device).eval()
    net.set_epoch(50)
    for k, v in old.state_dict().items():
        assert torch.equal(v, net.state_dict()[k]), k
    image = torch.randn(2, 4, *config.patch_size, device=device)
    with torch.no_grad():
        legacy, current = old(image), net(image, modality_mask=torch.ones(2, 4, device=device))
    for u, v in zip(legacy, current):
        torch.testing.assert_close(u, v, atol=2e-6, rtol=1e-5)
    checks['legacy_full_input_initialization_and_outputs'] = True
    del old, net, legacy, current
    gc.collect()
    seed_experiment(0)
    net = build_agent_network(config, communication=True, lomix=True).to(device)
    net.set_epoch(120)
    net.eval()
    with torch.no_grad():
        net.communication.rho.fill_(.1)
    hits = [0] * 4
    def hook(m):
        def observe(module, args, output):
            hits[m] += args[0].shape[0]
        return observe
    handles = [e.register_forward_hook(hook(m)) for m, e in enumerate(net.semantic.experts)]
    with torch.no_grad():
        for code in MASK_CODES:
            hits[:] = [0] * 4
            mask = mask_from_code(code, device=device)[None]
            parts = net(image[:1], True, modality_mask=mask)
            weights = parts['weights']
            assert torch.isfinite(parts['final']).all()
            assert torch.equal(weights.masked_select(~mask[:, None, :]), torch.zeros_like(weights.masked_select(~mask[:, None, :])))
            torch.testing.assert_close(weights.sum(-1), torch.ones(1, 3, device=device))
            assert hits == mask.long()[0].tolist(), (code, hits)
            if code in (1, 2, 4, 8):
                assert torch.count_nonzero(parts['attention']) == 0
        mixed = torch.tensor([[1, 1, 0, 0], [0, 0, 1, 0]], dtype=torch.bool, device=device)
        a = net(image, modality_mask=mixed)[0]
        for b in range(2):
            z = net(image[b:b+1], modality_mask=mixed[b:b+1])[0]
            torch.testing.assert_close(a[b:b+1], z, atol=3e-5, rtol=3e-4)
        hidden = image.clone().masked_fill(~mixed[:, :, None, None, None], 1e3)
        b = net(hidden, modality_mask=mixed)[0]
        torch.testing.assert_close(a, b, rtol=0, atol=0)
        hits[:] = [0] * 4
        net(torch.zeros_like(image[:1]), modality_mask=mask_from_code(1, device=device)[None])
        assert hits == [1, 0, 0, 0]
        full = torch.ones(1, 4, dtype=torch.bool, device=device)
        out_ds = net(image[:1], modality_mask=full)[0]
        net.decoder.deep_supervision = False
        out = net(image[:1], modality_mask=full)
        torch.testing.assert_close(out, out_ds)
        net.decoder.deep_supervision = True
    for h in handles:
        h.remove()
    checks['all15_sparse_calls_zero_background_hidden_invariance_mixed_batch'] = True
    checks['deep_supervision_off_preserves_final_output'] = True
    net.train()
    optimizer = torch.optim.SGD(net.parameters(), lr=.001)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == 'cuda')
    with torch.no_grad():
        net.communication.rho.zero_()
    with torch.autocast(device_type=device.type, enabled=device.type == 'cuda'):
        parts = net(image, True, modality_mask=mixed)
        labels = torch.randint(0, 4, (2, *config.patch_size), device=device)
        target = torch.stack((labels > 0, labels > 1, labels > 2), 1).float()
        aux, _ = present_expert_loss(parts['local_experts'], target, mixed, positive_region_loss)
        mix, _ = net.lomix(parts['joint_outputs'], target)
        loss = positive_region_loss(parts['final'], target) + aux + .1 * mix
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    assert all(p.grad is None for p in net.semantic.experts[3].parameters())
    assert all(p.grad is None for p in net.communication.project[3].parameters())
    assert all(any(p.grad is not None and p.grad.abs().max() > 0 for p in net.semantic.experts[m].parameters()) for m in (0, 1, 2))
    assert net.communication.rho.grad.abs().max() > 0
    assert net.lomix.alpha[0].grad.abs().max() > 0 and net.lomix.loss_logits.grad.abs().max() > 0
    assert torch.isfinite(loss)
    scaler.step(optimizer)
    scaler.update()
    del parts, loss, aux, mix
    net.zero_grad(set_to_none=True)
    with torch.no_grad():
        net.communication.rho.fill_(.1)
    with torch.autocast(device_type=device.type, enabled=device.type == 'cuda'):
        parts = net(image[:1], True, modality_mask=torch.ones(1, 4, device=device))
        loss = positive_region_loss(parts['final'], target[:1])
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    assert net.communication.correct[0][-1].weight.grad.abs().max() > 0
    assert net.communication.query.weight.grad.abs().max() > 0
    checks['absent_no_gradient_present_gradient_comm_unlock_lomix_gradient'] = True
    checks['synthetic_patch'] = config.patch_size
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-root', type=Path, default=Path('/data02/chujingyi/data/nnUNet_preprocessed/Dataset137_BraTS2021'))
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    report = {'status': 'running', 'torch': torch.__version__, 'device': args.device}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    try:
        report['core'] = run_core(args.dataset_root, torch.device(args.device))
        report['status'] = 'passed'
    except BaseException:
        report['status'], report['error'] = 'failed', traceback.format_exc()
        raise
    finally:
        args.output.write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
