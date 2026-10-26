#!/usr/bin/env python
"""CPU regression checks for the independent plans-sized fifth expert.

Uses the real stage widths/topology with a smaller synthetic spatial patch;
never loads patient data, performs optimizer updates, or starts GPU training.
"""
import argparse
from copy import deepcopy
import gc
import hashlib
import inspect
import json
from pathlib import Path
import tempfile

import torch

import nnunetv2
from nnunetv2.utilities.find_class_by_name import recursive_find_python_class
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager, ConfigurationManager
from nnunetv2.training.nnUNetTrainer.variants.brats2021_audit.nnUNetTrainer_BraTSAgentGeneralistV2Loss300 import (
    nnUNetTrainer_BraTSAgentGeneralistLegacyV2Loss300 as Legacy,
)
from nnunetv2.training.nnUNetTrainer.variants.brats2021_audit.nnUNetTrainer_BraTSAgentGeneralistFullSizeLegacyV2Loss300 import (
    nnUNetTrainer_BraTSAgentGeneralistFullSizeLegacyV2Loss300 as FullSize,
)
from nnunetv2.utilities.brats2021_audit_v2.missing_modality import mask_from_code
from nnunetv2.utilities.brats2021_audit_v2.missing_predictor import MissingModalityPredictor
from nnunetv2.utilities.brats2021_audit_v2.agent_evaluate import aggregate_case_routes
from nnunetv2.utilities.brats2021_audit_v2.agent_v1.diagnostics import output_summary


def digest(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def parameter_counts(network):
    return {
        'total': sum(p.numel() for p in network.parameters()),
        'joint': sum(p.numel() for name, p in network.named_parameters()
                     if name.startswith(('encoder.', 'decoder.'))),
        'fifth': sum(p.numel() for p in network.semantic.experts[4].parameters()),
        'fifth_unused_heads': sum(p.numel() for p in
                                 (network.semantic.experts[4].decoder.seg_layers[:-1].parameters()
                                  if hasattr(network.semantic.experts[4], 'decoder') else
                                  network.semantic.experts[4].seg_layers[:-1].parameters())),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plans', type=Path, default=Path(
        '/data02/chujingyi/data/nnUNet_preprocessed/Dataset137_BraTS2021/nnUNetPlans.json'))
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    plans = PlansManager(str(args.plans))
    actual = plans.get_configuration('3d_fullres')
    report = {'plans': str(args.plans.resolve()), 'device': 'cpu', 'patient_data_used': False,
              'optimizer_updates': 0, 'actual_plan_parameter_counts': {}}
    for cls in (Legacy, FullSize):
        torch.manual_seed(0)
        model = cls.build_network_architecture(plans, actual, 4, 3)
        report['actual_plan_parameter_counts'][cls.__name__] = parameter_counts(model)
        del model
        gc.collect()

    configuration = deepcopy(actual.configuration)
    configuration['patch_size'] = [32, 64, 16]
    small = ConfigurationManager(configuration)
    torch.manual_seed(17)
    old = Legacy.build_network_architecture(plans, small, 4, 3)
    old_rng = torch.get_rng_state().clone()
    retained = {name: digest(value) for name, value in old.state_dict().items()
                if not name.startswith(('semantic.experts.4.', 'communication.project.4.'))
                and name != 'architecture_version'}
    del old
    gc.collect()
    torch.manual_seed(17)
    model = FullSize.build_network_architecture(plans, small, 4, 3)
    assert torch.equal(torch.get_rng_state(), old_rng), 'New branch changed external torch RNG'
    for name, expected in retained.items():
        assert digest(model.state_dict()[name]) == expected, name
    assert FullSize.network_options() == Legacy.network_options()
    assert FullSize.compute_agent_loss is Legacy.compute_agent_loss
    assert FullSize.train_step is Legacy.train_step
    assert tuple(model.semantic.experts[4].encoder.output_channels) == tuple(model.encoder.output_channels)
    assert model.communication.project[4].in_channels == model.semantic.experts[4].feature_channels
    assert all(not p.requires_grad for p in model.semantic.experts[4].decoder.seg_layers[:-1].parameters())
    assert not ({id(p) for p in model.encoder.parameters()} &
                {id(p) for p in model.semantic.experts[4].parameters()})
    report['retained_tensor_hashes_match_legacy'] = len(retained)
    report['torch_rng_unchanged'] = True
    report['same_loss_train_step_and_network_options'] = True

    x = torch.randn(1, 4, *small.patch_size)
    expert = model.semantic.experts[4]
    expert_input = torch.cat((x, torch.ones_like(x)), 1)
    with torch.no_grad():
        plain = expert(expert_input)
        exposed, features = expert(expert_input, return_features=True)
        assert torch.equal(plain, exposed), 'Feature capture changed official decoder output'
        assert features.shape[1] == expert.feature_channels
    assert not expert.decoder.stages[expert.feature_stage]._forward_hooks
    del plain, exposed, features, expert_input

    for epoch, fusion, communication in ((0, False, False), (50, True, False),
                                         (100, True, True), (299, True, True)):
        model.set_epoch(epoch)
        assert bool(model.semantic.fusion_active) == fusion
        assert bool(model.communication_active) == communication
        assert all(not p.requires_grad for p in expert.decoder.seg_layers[:-1].parameters()), epoch
    report['unused_coarse_heads_frozen_at_epochs'] = [0, 50, 100, 299]
    model.set_epoch(100)
    model.train()
    checks = []
    for code in range(1, 16):
        mask = mask_from_code(code)[None]
        model.zero_grad(set_to_none=True)
        parts = model(x, return_aux=True, modality_mask=mask)
        assert parts['final'].shape == (1, 3, *small.patch_size)
        assert torch.isfinite(parts['final']).all()
        assert parts['expert_mask'].shape == (1, 5) and parts['expert_mask'][0, 4]
        assert torch.all(parts['weights'][:, :, :4][:, :, ~mask[0]] == 0)
        assert torch.allclose(parts['weights'].sum(-1), torch.ones(1, 3), atol=1e-6)
        assert torch.all(parts['attention'].diagonal(dim1=1, dim2=3) == 0)
        # A synthetic objective exercises all experts and all joint DS heads.
        objective = sum(v.square().mean() for v in parts['output'])
        objective = objective + sum(v.square().mean() for v in parts['local_experts'])
        objective.backward()
        gradients = [p.grad for p in model.parameters() if p.grad is not None]
        assert gradients and all(torch.isfinite(g).all() for g in gradients)
        assert any(p.grad is not None and torch.count_nonzero(p.grad)
                   for p in expert.parameters()), 'Fifth expert receives no gradient'
        assert not expert.decoder.stages[expert.feature_stage]._forward_hooks
        record = output_summary(parts)
        routes = [{'case_id': 'synthetic', 'fold_index': 0, 'tile': 0, 'tta': 0, **record}]
        aggregate_case_routes(routes, 'synthetic', code)
        checks.append(code)
        del parts, objective, gradients, record, routes
        print('Checked forward/backward/masked routing:', code, flush=True)
    report['all15_forward_backward_routing'] = checks

    model.zero_grad(set_to_none=True)
    model.eval()
    model.decoder.deep_supervision = False
    with torch.inference_mode():
        mask = mask_from_code(3)[None]
        expected = model(x, modality_mask=mask)
        dirty = x.clone()
        dirty[:, ~mask[0]] = 1000
        assert torch.equal(expected, model(dirty, modality_mask=mask)), 'Missing channels leaked into predictions'
    # Exercise the exact predictor's mask dispatch and five-expert route export.
    predictor = MissingModalityPredictor(device=torch.device('cpu'), use_mirroring=False,
                                        perform_everything_on_device=False, allow_tqdm=False)
    predictor.network = model
    predictor.configuration_manager = small
    predictor.allowed_mirroring_axes = None
    predictor.list_of_parameters = None
    prediction, routes = predictor.predict_with_mask(x[0], mask[0], collect_routes=True)
    assert predictor._supports_mask
    assert torch.allclose(prediction, expected[0], atol=1e-5)
    aggregate_case_routes(routes, 'case', 3)
    assert 'modality_mask' in inspect.signature(model.forward).parameters
    report['existing_missing_predictor_and_route_aggregation'] = 'pass'

    with tempfile.TemporaryDirectory(prefix='generalist_fullsize_check_') as folder:
        path = Path(folder) / 'network.pth'
        torch.save(model.state_dict(), path)
        del predictor, model, expert
        gc.collect()
        restored = FullSize.build_network_architecture(plans, small, 4, 3, enable_deep_supervision=False)
        restored.load_state_dict(torch.load(path, map_location='cpu', weights_only=False), strict=True)
        restored.restore_stage()
        assert int(restored.stage_epoch) == 100 and bool(restored.communication_active)
        restored.eval()
        with torch.inference_mode():
            assert torch.equal(expected, restored(x, modality_mask=mask))
    report['strict_checkpoint_roundtrip'] = 'pass'
    registry = str(Path(nnunetv2.__path__[0]) / 'training/nnUNetTrainer')
    found = recursive_find_python_class(registry, FullSize.__name__, 'nnunetv2.training.nnUNetTrainer')
    assert found is FullSize
    report['nnunet_trainer_discovery'] = FullSize.__name__
    report['gpu_memory_or_real_training_tested'] = False
    report['status'] = 'pass'
    output = json.dumps(report, indent=2)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(output + '\n')
    print(output)


if __name__ == '__main__':
    main()
