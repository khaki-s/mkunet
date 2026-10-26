"""Synthetic CPU checks; never starts patient training or writes result folders."""
import argparse
import copy
import gc
import json
import os
from pathlib import Path
import tempfile
import traceback

# Set before nnU-Net imports its path constants.
_WORK = tempfile.TemporaryDirectory(prefix='brats_feature_checks_')
os.environ['nnUNet_results'] = str(Path(_WORK.name) / 'results')
os.environ.setdefault('nnUNet_raw', '/data02/chujingyi/data/nnUNet_raw')
os.environ.setdefault('nnUNet_preprocessed', '/data02/chujingyi/data/nnUNet_preprocessed')
os.environ['nnUNet_wandb_enabled'] = 'false'
os.environ.setdefault('MPLCONFIGDIR', '/tmp/brats_feature_matplotlib')
os.environ['BRATS_DIAG_EVERY'] = '1'
os.environ['BRATS_ABLATE_EVERY'] = '0'

import numpy as np
import torch
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
from nnunetv2.utilities.find_objects import recursive_find_trainer_class_by_name
from nnunetv2.utilities.collate_outputs import collate_outputs
from nnunetv2.training.nnUNetTrainer.variants.brats2021_audit.nnUNetTrainer_BraTSAgentGeneralistFeature300 import (
    nnUNetTrainer_BraTSAgentGeneralistFeatureLocal300,
    nnUNetTrainer_BraTSAgentGeneralistFeatureComm300,
)
from ...agent_integration_check import batch_for
from ...missing_modality import MASK_CODES, mask_from_code
from ...reproducibility import seed_experiment, state_digest
from ..generalist.network import build_generalist_network
from .network import build_feature_injection_network
from .blocks import JointExpertCrossAttention3D


CLASSES = (nnUNetTrainer_BraTSAgentGeneralistFeatureLocal300,
           nnUNetTrainer_BraTSAgentGeneralistFeatureComm300)


def block_checks():
    seed_experiment(0)
    block = JointExpertCrossAttention3D(8, 6)
    joint = torch.randn(2, 8, 2, 3, 2, requires_grad=True)
    states = [torch.randn(2, 6, 2, 3, 2, requires_grad=True) for _ in range(5)]
    mask = torch.tensor([[1, 0, 0, 0, 1], [0, 1, 1, 0, 1]], dtype=torch.bool)
    first = block(joint, states, mask, capture=True)
    torch.testing.assert_close(first, joint, rtol=0, atol=0)
    first.square().mean().backward()
    assert block.output.weight.grad.abs().sum() > 0
    assert all(s.grad is None or s.grad.abs().sum() == 0 for s in states)
    with torch.no_grad():
        block.output.weight -= .1 * block.output.weight.grad
    block.zero_grad(set_to_none=True)
    joint.grad = None
    for s in states:
        s.grad = None
    second = block(joint, states, mask, capture=True)
    second.square().mean().backward()
    assert joint.grad.abs().sum() > 0
    assert block.attention.in_proj_weight.grad.abs().sum() > 0
    for m in range(5):
        if mask[:, m].any():
            assert states[m].grad[mask[:, m]].abs().sum() > 0
            if (~mask[:, m]).any():
                assert states[m].grad[~mask[:, m]].abs().sum() == 0
        else:
            assert states[m].grad is None or states[m].grad.abs().sum() == 0
    weights = block.last_attention
    assert torch.count_nonzero(weights.masked_select(~mask[:, None, :, None, None, None])) == 0
    torch.testing.assert_close(weights.sum(2), torch.ones_like(weights.sum(2)), rtol=1e-6, atol=1e-6)
    perturbed = [s.detach().clone() for s in states]
    perturbed[0][0] += torch.randn_like(perturbed[0][0])
    with torch.no_grad():
        changed = block(joint.detach(), perturbed, mask)
        assert not torch.equal(changed[0], second[0])
        torch.testing.assert_close(changed[1], second[1], rtol=1e-5, atol=1e-6)
        perturbed[3] += 1000
        invisible = block(joint.detach(), perturbed, mask)
        torch.testing.assert_close(changed, invisible, rtol=0, atol=0)
    return dict(zero_residual_identity=True, first_step_output_gradient=True,
                subsequent_attention_expert_and_joint_gradients=True,
                absent_expert_gradient_zero=True, masked_attention_exact_zero=True,
                present_state_changes_joint_feature=True, absent_state_invariant=True)


def initial_equality_checks(config):
    records = []
    for cls in CLASSES:
        options = cls.network_options()
        seed_experiment(0)
        old = build_generalist_network(config, **options).eval()
        seed_experiment(0)
        new = build_feature_injection_network(config, **options).eval()
        old_state, new_state = old.state_dict(), new.state_dict()
        for name, value in old_state.items():
            if name != 'architecture_version':
                torch.testing.assert_close(value, new_state[name], rtol=0, atol=0)
        assert set(new_state) - set(old_state) == {
            k for k in new_state if k.startswith('feature_injection.') or k == 'feature_injection_active'}
        image = torch.randn(1, 4, *config.patch_size)
        for epoch in (0, 50, 100):
            old.set_epoch(epoch); new.set_epoch(epoch)
            maximum = 0.
            for code in MASK_CODES:
                mask = mask_from_code(code)[None]
                with torch.no_grad():
                    a = old(image, True, modality_mask=mask)
                    b = new(image, True, modality_mask=mask)
                for lhs, rhs in zip(a['output'], b['output']):
                    torch.testing.assert_close(lhs, rhs, rtol=1e-5, atol=2e-6)
                    maximum = max(maximum, float((lhs - rhs).abs().max()))
                if epoch >= 50:
                    weights = b['feature_attention']
                    assert weights.shape[2] == 5
                    assert torch.count_nonzero(weights.masked_select(
                        ~b['expert_mask'][:, None, :, None, None, None])) == 0
                    torch.testing.assert_close(weights.sum(2), torch.ones_like(weights.sum(2)),
                                               rtol=1e-6, atol=1e-6)
                assert b['expert_mask'][:, 4].all()
            print(f'initial equality {cls.__name__} epoch={epoch}: {maximum}', flush=True)
            records.append(dict(trainer=cls.__name__, epoch=epoch, masks=15, maximum_difference=maximum))
        # One pass per decoder stage, and no execution of missing specialists.
        new.set_epoch(100)
        counters = [0] * 5
        decoders = [0] * len(new.decoder.stages)
        handles = []
        def count(target, idx):
            def hook(module, inputs, output):
                target[idx] += 1
            return hook
        for i, expert in enumerate(new.semantic.experts):
            handles.append(expert.register_forward_hook(count(counters, i)))
        for i, stage in enumerate(new.decoder.stages):
            handles.append(stage.register_forward_hook(count(decoders, i)))
        with torch.no_grad():
            out = new(image, True, modality_mask=mask_from_code(1)[None])
        for handle in handles:
            handle.remove()
        assert counters == [1, 0, 0, 0, 1] and decoders == [1] * len(decoders)
        hidden = image.clone(); hidden[:, 1:] = 9999.
        with torch.no_grad():
            hidden_output = new(hidden, True, modality_mask=mask_from_code(1)[None])
        torch.testing.assert_close(out['final'], hidden_output['final'], rtol=0, atol=0)
        try:
            new(image, modality_mask=torch.zeros(1, 4, dtype=torch.bool))
        except ValueError:
            pass
        else:
            raise AssertionError('All-missing input accepted')
        del old, new, a, b, image, hidden, hidden_output, old_state, new_state, out
        gc.collect()
    return dict(initial_same_seed_parameters=True, same_old_parameter_names=True,
                all15_phase_equality=records, one_joint_decoder_pass=True,
                absent_specialists_not_executed=True, hidden_channel_invariance=True,
                all_missing_rejected=True)


def trainer_checks(plans, dataset):
    records = []
    for cls in CLASSES:
        assert recursive_find_trainer_class_by_name(cls.__name__) is cls
        def make():
            trainer = cls(copy.deepcopy(plans), '3d_fullres', 0, dataset, device=torch.device('cpu'))
            trainer.num_iterations_per_epoch = 1
            trainer.initialize()
            assert Path(trainer.output_folder).is_relative_to(Path(_WORK.name))
            assert trainer.num_epochs == trainer.lr_scheduler.max_steps == 300
            return trainer
        trainer = make()
        ids = {id(p) for group in trainer.optimizer.param_groups for p in group['params']}
        assert {id(p) for p in trainer.network.feature_injection.parameters()}.issubset(ids)
        batch = batch_for(trainer, 1)
        # Exercise actual loss/backward/optimizer and inherited epoch-end collator.
        for epoch in (0, 50, 100):
            # We jump directly to phase boundaries; supply synthetic prior
            # logger history only (no skipped-epoch training is claimed).
            for key, history in trainer.logger.local_logger.my_fantastic_logging.items():
                while len(history) < epoch:
                    history.append(copy.deepcopy(history[-1]) if history else
                                   ([0., 0., 0.] if key == 'dice_per_class_or_region' else 0.))
            trainer.current_epoch = epoch
            trainer._agent_steps = epoch
            trainer.on_train_epoch_start()
            trainer.network.train()
            result = trainer.train_step(batch)
            assert np.isfinite(result['loss'])
            collate_outputs([result])
            trainer.on_train_epoch_end([result])
            trainer.on_validation_epoch_start()
            with torch.no_grad():
                value = trainer.validation_step(batch)
            trainer.on_validation_epoch_end([value])
            assert np.isfinite(value['loss'])
            print(f'actual train/val epoch hooks {cls.__name__} epoch={epoch}', flush=True)
            records.append(dict(trainer=cls.__name__, epoch=epoch, loss=float(result['loss']),
                                train_and_val_epoch_end=True))
        checkpoint = Path(_WORK.name) / f'{cls.__name__}.pth'
        trainer.save_checkpoint(checkpoint)
        resumed = make()
        resumed.load_checkpoint(checkpoint)
        assert int(resumed.network.stage_epoch) == 100
        assert bool(resumed.network.feature_injection_active)
        assert state_digest(trainer.network) == state_digest(resumed.network)
        trainer.current_epoch = 101
        for current in (trainer, resumed):
            current.on_train_epoch_start()
            current.train_step(batch)
        assert state_digest(trainer.network) == state_digest(resumed.network)
        # Isolate the new expert->joint path from final logit fusion/aux losses.
        trainer.network.eval()
        trainer.optimizer.zero_grad(set_to_none=True)
        mask = mask_from_code(1)[None]
        parts = trainer.network(batch['data'], True, modality_mask=mask)
        parts['joint'].square().mean().backward()
        assert trainer.network.feature_injection.attention.in_proj_weight.grad.abs().sum() > 0
        for m in (0, 4):
            assert sum(float(p.grad.abs().sum()) for p in trainer.network.semantic.experts[m].parameters()
                       if p.grad is not None) > 0
        assert sum(float(p.grad.abs().sum()) for p in trainer.network.encoder.parameters()
                   if p.grad is not None) > 0
        with torch.no_grad():
            excluded = trainer.network(batch['data'], True, modality_mask=mask,
                                       disable_feature_injection=True)
        assert not torch.equal(parts['joint'], excluded['joint'])
        print(f'checkpoint+resume+isolated joint gradients {cls.__name__}: passed', flush=True)
        del trainer, resumed, batch, parts, excluded
        gc.collect()
    return dict(discoverable=[c.__name__ for c in CLASSES], parameters_in_optimizer=True,
                actual_training_phases=records, strict_checkpoint_restore=True,
                next_resumed_step_bitwise_equal=True, expert_to_joint_gradient_path=True,
                trained_injection_changes_joint=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-root', type=Path,
                        default=Path('/data02/chujingyi/data/nnUNet_preprocessed/Dataset137_BraTS2021'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    report = dict(status='running', scope='synthetic CPU small plans, no patient training or GPU measurements',
                  torch=torch.__version__)
    plans = json.loads((args.dataset_root / 'nnUNetPlans.json').read_text())
    dataset = json.loads((args.dataset_root / 'dataset.json').read_text())
    plans['continue_training'] = False
    config = plans['configurations']['3d_fullres']
    config['patch_size'], config['batch_size'] = [64, 64, 32], 1
    config['architecture']['arch_kwargs']['features_per_stage'] = [8, 16, 32, 64, 80, 80]
    try:
        report['block'] = block_checks()
        report['network'] = initial_equality_checks(PlansManager(plans).get_configuration('3d_fullres'))
        report['trainer'] = trainer_checks(plans, dataset)
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
