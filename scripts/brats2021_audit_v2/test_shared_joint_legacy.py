#!/usr/bin/env python
"""CPU checks for a shared joint node; no patient images or optimizer updates."""
import argparse
from copy import deepcopy
import gc
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace

import torch

import nnunetv2
from nnunetv2.utilities.find_class_by_name import recursive_find_python_class
from nnunetv2.utilities.collate_outputs import collate_outputs
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager, ConfigurationManager
from nnunetv2.training.nnUNetTrainer.variants.brats2021_audit.nnUNetTrainer_BraTSAgentGeneralistV2Loss300 import (
    nnUNetTrainer_BraTSAgentGeneralistLegacyV2Loss300 as Legacy,
)
from nnunetv2.training.nnUNetTrainer.variants.brats2021_audit.nnUNetTrainer_BraTSAgentSharedJointLegacyV2Loss300 import (
    nnUNetTrainer_BraTSAgentSharedJointLegacyV2Loss300 as SharedJoint,
)
from nnunetv2.utilities.brats2021_audit_v2.agent_v2.shared_joint.diagnostics import (
    SharedJointRoutingAccumulator, shared_joint_summary,
)
from nnunetv2.utilities.brats2021_audit_v2.agent_v1.diagnostics import output_summary
from nnunetv2.utilities.brats2021_audit_v2.missing_modality import mask_from_code
from nnunetv2.utilities.brats2021_audit_v2.missing_predictor import MissingModalityPredictor
from nnunetv2.utilities.brats2021_audit_v2.agent_evaluate import aggregate_case_routes


def digest(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def counts(model):
    return {'total': sum(p.numel() for p in model.parameters()),
            'joint': sum(p.numel() for name, p in model.named_parameters()
                         if name.startswith(('encoder.', 'decoder.'))),
            'independent_expert_modules': len(model.semantic.experts),
            'expert_parameters': sum(p.numel() for p in model.semantic.experts.parameters()),
            'communication_parameters': sum(p.numel() for p in model.communication.parameters())}


def parameter_aliases(model):
    groups = {}
    for name, value in model.named_parameters(remove_duplicate=False):
        groups.setdefault(id(value), []).append(name)
    return sorted(sorted(names) for names in groups.values() if len(names) > 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plans', type=Path, default=Path(
        '/data02/chujingyi/data/nnUNet_preprocessed/Dataset137_BraTS2021/nnUNetPlans.json'))
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    plans = PlansManager(str(args.plans))
    actual = plans.get_configuration('3d_fullres')
    report = {'patient_data_used': False, 'device': 'cpu', 'optimizer_updates': 0,
              'actual_plan_parameters': {}}
    for cls in (Legacy, SharedJoint):
        torch.manual_seed(0)
        model = cls.build_network_architecture(plans, actual, 4, 3)
        report['actual_plan_parameters'][cls.__name__] = counts(model)
        print('Actual plans:', cls.__name__, report['actual_plan_parameters'][cls.__name__], flush=True)
        del model
        gc.collect()
    configuration = deepcopy(actual.configuration)
    configuration['patch_size'] = [32, 64, 16]
    small = ConfigurationManager(configuration)
    torch.manual_seed(17)
    old = Legacy.build_network_architecture(plans, small, 4, 3)
    rng = torch.get_rng_state().clone()
    old_aliases = parameter_aliases(old)
    # Fifth node-specific modules intentionally change; all main/four-specialist
    # tensors and shared QKV are retained from the established constructor.
    excluded = ('semantic.experts.4.', 'communication.project.4.',
                'communication.correct.4.', 'communication.gate.')
    retained = {name: digest(value) for name, value in old.state_dict().items()
                if not name.startswith(excluded) and name != 'architecture_version'}
    del old
    gc.collect()
    torch.manual_seed(17)
    model = SharedJoint.build_network_architecture(plans, small, 4, 3)
    assert torch.equal(torch.get_rng_state(), rng)
    for name, expected in retained.items():
        assert digest(model.state_dict()[name]) == expected, name
    report['retained_legacy_tensor_hashes_match'] = len(retained)
    report['external_torch_rng_matches'] = True
    assert len(model.semantic.experts) == 4
    assert not any(name.startswith('semantic.experts.4.') for name in model.state_dict())
    all_names = list(model.named_parameters(remove_duplicate=False))
    # Official PlainConvUNet already stores encoder under decoder.encoder.
    # Keep those inherited aliases, introduce none for the shared communication node.
    assert parameter_aliases(model) == old_aliases
    optimizer_parameters = list(model.parameters())
    assert len(optimizer_parameters) == len({id(value) for value in optimizer_parameters})
    report['no_extra_fifth_expert_or_new_parameter_aliases'] = True
    report['inherited_official_parameter_alias_groups'] = len(old_aliases)
    report['optimizer_parameter_objects_unique'] = True
    assert SharedJoint.train_step is Legacy.train_step
    assert SharedJoint.network_options() == Legacy.network_options()
    for epoch, fusion, communication in ((0, False, False), (50, True, False), (100, True, True), (299, True, True)):
        model.set_epoch(epoch)
        assert bool(model.semantic.fusion_active) == fusion
        assert bool(model.communication_active) == communication
    report['stage_schedule'] = [0, 50, 100]

    # Warmup supervision counts the raw joint only in main, not local expert aux.
    toy = [torch.tensor(float(i + 1), requires_grad=True).reshape(1, 1, 1, 1, 1) for i in range(5)]
    dummy = SimpleNamespace(loss=lambda outputs, target: outputs[0].mean(),
        region_aux_loss=lambda logits, target: logits.mean(),
        network=SimpleNamespace(stage_epoch=0, communication_active=False),
        refined_aux_max=.1, refined_aux_ramp_epochs=20)
    parts = {'output': [toy[4]], 'local_experts': toy, 'modality_mask': torch.ones(1, 4, dtype=torch.bool),
             'active': False, 'moe': toy[4]}
    loss, values = SharedJoint.compute_agent_loss(dummy, parts, torch.zeros_like(toy[4]))
    assert float(loss) == 7.5 and values['expert_mean'] == 2.5
    assert values['expert_losses'] == [1., 2., 3., 4., None]
    assert float(torch.autograd.grad(loss, toy[4])[0]) == 1.
    collate_outputs([dict(loss=1., **values), dict(loss=1., **values)])
    report['warmup_joint_loss_counted_once_and_epoch_collation'] = 'pass'

    x = torch.randn(1, 4, *small.patch_size)
    model.train()
    model.set_epoch(0)
    warm = model(x, True)
    joint_parameter = next(model.encoder.parameters())
    specialist_loss = sum(z.square().mean() for z in warm['local_experts'][:4])
    assert torch.autograd.grad(specialist_loss, joint_parameter, allow_unused=True)[0] is None
    assert warm['local_experts'][4] is warm['joint'] and warm['final'] is warm['joint']
    del warm, specialist_loss
    model.set_epoch(100)
    complete = []
    for code in range(1, 16):
        calls = {'encoder': 0, 'decoder': 0}
        def count(which):
            def hook(module, inputs, outputs):
                calls[which] += 1
            return hook
        handles = [model.encoder.register_forward_hook(count('encoder')),
                   model.decoder.register_forward_hook(count('decoder'))]
        model.zero_grad(set_to_none=True)
        mask = mask_from_code(code)[None]
        parts = model(x, True, modality_mask=mask)
        assert parts['local_experts'][4] is parts['joint']
        assert parts['states'][4] is parts['joint_state']
        assert parts['joint_state'].requires_grad and parts['joint'].requires_grad
        assert parts['final'].shape == (1, 3, *small.patch_size)
        assert torch.isfinite(parts['final']).all()
        assert torch.all(parts['weights'][:, :, :4][:, :, ~mask[0]] == 0)
        assert torch.allclose(parts['weights'].sum(-1), torch.ones(1, 3), atol=1e-6)
        assert parts['expert_mask'].shape == (1, 5) and parts['expert_mask'][0, 4]
        assert torch.all(parts['attention'].diagonal(dim1=1, dim2=3) == 0)
        objective = sum(z.square().mean() for z in parts['output'])
        objective = objective + sum(z.square().mean() for z in parts['local_experts'][:4])
        objective.backward()
        assert calls == {'encoder': 1, 'decoder': 1}, calls
        for handle in handles:
            handle.remove()
        assert not model.decoder.stages[model.joint_feature_stage]._forward_hooks
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        assert torch.count_nonzero(joint_parameter.grad) > 0
        record = output_summary(parts)
        rows = aggregate_case_routes([{'case_id':'synthetic', **record}], 'synthetic', code)
        assert len(rows) == 15 and {r['modality'] for r in rows} == {'T1', 'T1ce', 'T2', 'FLAIR', 'Joint'}
        accumulator = SharedJointRoutingAccumulator()
        accumulator.add(parts)
        assert accumulator.finish()[str(code)]['expert_names'][-1] == 'Joint'
        summary = shared_joint_summary(model, parts)
        assert 'expert_Joint' not in summary['gradients']
        complete.append(code)
        del objective, parts, record, rows, summary
        print('Checked single joint pass and masked routes:', code, flush=True)
    report['all15_single_joint_forward_backward_routes'] = complete

    model.zero_grad(set_to_none=True)
    mixed_mask = torch.stack((mask_from_code(1), mask_from_code(8)))
    mixed = model(torch.cat((x, x * 1.2), dim=0), True, modality_mask=mixed_mask)
    assert mixed['final'].shape[0] == 2
    assert torch.all(mixed['weights'][:, :, :4].masked_select(~mixed_mask[:, None].expand(-1, 3, -1)) == 0)
    assert torch.all(mixed['expert_mask'][:, 4])
    mixed['final'].square().mean().backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    del mixed
    report['heterogeneous_two_sample_masks'] = 'pass'

    # Zero-initialized correction outputs intentionally make the first forward
    # identity. Nonzero test heads expose BOTH learned communication directions.
    with torch.no_grad():
        for head in model.communication.correct:
            torch.nn.init.normal_(head[-1].weight, std=.02)
    model.zero_grad(set_to_none=True)
    parts = model(x, True)
    direct_grad = torch.autograd.grad(parts['joint'].square().mean(), joint_parameter, retain_graph=True)[0]
    mixture_grad = torch.autograd.grad(parts['moe'].square().mean(), joint_parameter, retain_graph=True)[0]
    sent_objective = sum((parts['experts'][i] - parts['local_experts'][i]).square().mean() for i in range(4))
    sent_grad = torch.autograd.grad(sent_objective, joint_parameter, retain_graph=True)[0]
    received = (parts['experts'][4] - parts['joint']).square().mean()
    received_grad = torch.autograd.grad(received, next(model.semantic.experts[0].parameters()))[0]
    for value in (direct_grad, mixture_grad, sent_grad, received_grad):
        assert torch.isfinite(value).all() and torch.count_nonzero(value) > 0
    report['joint_live_gradients'] = {'direct_bypass': True, 'five_node_mixture': True,
                                    'joint_sends_to_specialists': True, 'joint_receives_from_specialists': True}
    del parts, direct_grad, mixture_grad, sent_grad, received_grad, sent_objective, received
    model.zero_grad(set_to_none=True)
    model.eval()
    model.decoder.deep_supervision = False
    mask = mask_from_code(3)[None]
    with torch.inference_mode():
        expected = model(x, modality_mask=mask)
        dirty = x.clone()
        dirty[:, ~mask[0]] = 1234
        assert torch.equal(expected, model(dirty, modality_mask=mask))
    predictor = MissingModalityPredictor(device=torch.device('cpu'), use_mirroring=False,
        perform_everything_on_device=False, allow_tqdm=False)
    predictor.network, predictor.configuration_manager = model, small
    predictor.allowed_mirroring_axes = None
    predictor.list_of_parameters = None
    prediction, routes = predictor.predict_with_mask(x[0], mask[0], collect_routes=True)
    assert torch.allclose(prediction, expected[0], atol=1e-5)
    assert any(row['modality'] == 'Joint' for row in aggregate_case_routes(routes, 'case', 3))
    report['existing_missing_predictor'] = 'pass'
    with tempfile.TemporaryDirectory(prefix='shared_joint_check_') as directory:
        file = Path(directory) / 'network.pth'
        torch.save(model.state_dict(), file)
        del model, predictor, joint_parameter
        gc.collect()
        restored = SharedJoint.build_network_architecture(plans, small, 4, 3, False)
        restored.load_state_dict(torch.load(file, map_location='cpu', weights_only=False), strict=True)
        restored.restore_stage()
        assert int(restored.stage_epoch) == 100
        restored.eval()
        with torch.inference_mode():
            assert torch.equal(expected, restored(x, modality_mask=mask))
    found = recursive_find_python_class(str(Path(nnunetv2.__path__[0]) / 'training/nnUNetTrainer'),
                                       SharedJoint.__name__, 'nnunetv2.training.nnUNetTrainer')
    assert found is SharedJoint
    report['strict_checkpoint_roundtrip_and_trainer_discovery'] = 'pass'
    report['real_gpu_memory_or_training_tested'] = False
    report['status'] = 'pass'
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
