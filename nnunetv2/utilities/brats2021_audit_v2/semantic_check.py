"""Bounded synthetic checks; --gpu additionally checks the actual planned patch.

Never trains on dataset images or writes experiment checkpoints. Synthetic
optimizer steps and restore files live under a TemporaryDirectory. No change
to the production GPU logging/training path is needed for these checks.
"""
import argparse
import ast
import copy
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import tempfile

import numpy as np
import torch


class RandomBatchFixture:
    def set_thread_id(self, thread_id):
        pass

    def __next__(self):
        return {'data': torch.rand(3), 'numpy': np.random.rand(3), 'python': random.random()}


def batch_signature(batch):
    return (batch['data'].tolist(), batch['numpy'].tolist(), batch['python'])


def run(args):
    os.environ['nnUNet_raw'] = '/root/autodl-tmp/data/nnUNet_raw'
    os.environ['nnUNet_preprocessed'] = '/root/autodl-tmp/data/nnUNet_preprocessed'
    os.environ['nnUNet_wandb_enabled'] = 'False'
    os.environ['nnUNet_compile'] = 'False'
    os.environ['BRATS_AUDIT_SEED'] = '0'
    os.environ['BRATS_DIAG_EVERY'] = '1'
    torch.set_num_threads(4)
    with tempfile.TemporaryDirectory(prefix='brats_semantic_check_') as tmp:
        os.environ['nnUNet_results'] = tmp
        from . import vendor
        from .reproducibility import seed_experiment, state_digest, seeded_augmenter
        from .semantic_diagnostics import semantic_probe
        from nnunetv2.utilities.find_objects import recursive_find_trainer_class_by_name
        from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
        from nnunetv2.training.nnUNetTrainer.variants.brats2021_audit import nnUNetTrainer_BraTSAudit200 as registry
        source = Path('/root/brats_dataset_audit/reference_code/SimMLM-main/models/nnunet.py')
        vendored = Path(vendor.__file__).parent / 'simmlm_unet.py'
        extract = lambda p: next(x for x in ast.parse(p.read_text()).body if isinstance(x, ast.ClassDef) and x.name == 'UNet')
        assert ast.dump(extract(source)) == ast.dump(extract(vendored))
        root = Path(os.environ['nnUNet_preprocessed']) / 'Dataset137_BraTS2021'
        plans = json.loads((root / 'nnUNetPlans.json').read_text())
        dataset = json.loads((root / 'dataset.json').read_text())
        classes = [registry.nnUNetTrainer_BraTSAuditBaseline300, registry.nnUNetTrainer_BraTSExclusiveLoss300,
                   registry.nnUNetTrainer_BraTSSemanticMoELoss300, registry.nnUNetTrainer_BraTSSemanticMoEUniformLoss300]
        report = {'unmodified_official_UNet_class': True, 'initialization': {}, 'training_contracts': {}, 'gpu': 'not run'}
        manager = PlansManager(plans)
        config = manager.get_configuration('3d_fullres')
        for cls in classes:
            assert recursive_find_trainer_class_by_name(cls.__name__) is cls
            seed_experiment(0)
            net = cls.build_network_architecture(manager, config, 4, 3, True)
            row = {'encoder': state_digest(net.encoder), 'decoder': state_digest(net.decoder),
                   'rng_after': hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest()}
            if hasattr(net, 'semantic'):
                row['experts'] = [state_digest(x) for x in net.semantic.experts]
            report['initialization'][cls.__name__] = row
            del net
            gc.collect()
        rows = list(report['initialization'].values())
        for key in ('encoder', 'decoder', 'rng_after'):
            assert len({r[key] for r in rows}) == 1, key
        assert rows[2]['experts'] == rows[3]['experts']
        seed_experiment(1)
        changed = classes[0].build_network_architecture(manager, config, 4, 3, True)
        assert state_digest(changed.encoder) != rows[0]['encoder']
        del changed
        gc.collect()

        sequences = []
        for workers, offset in ((0, 0), (2, 0), (2, 3)):
            aug = seeded_augmenter(RandomBatchFixture(), 0, 0, workers, offset, False)
            try:
                sequences.append([batch_signature(next(aug)) for _ in range(6 - offset)])
            finally:
                if hasattr(aug, '_finish'):
                    aug._finish()
        assert sequences[0] == sequences[1]
        assert sequences[0][3:] == sequences[2]
        report['seeded_python_numpy_torch_batches_and_resume_offset'] = True

        for cls in classes[2:]:
            small = copy.deepcopy(plans)
            small['continue_training'] = False
            cfg = small['configurations']['3d_fullres']
            cfg['patch_size'], cfg['batch_size'] = [64, 64, 32], 1
            # Reduce ONLY the synthetic joint backbone, retaining real expert widths.
            cfg['architecture']['arch_kwargs']['features_per_stage'] = [8, 16, 32, 64, 80, 80]
            trainer = cls(small, '3d_fullres', 0, dataset, device=torch.device('cpu'))
            trainer.initialize()
            assert trainer.num_epochs == 300
            assert trainer.exclusive_loss
            assert trainer.lr_scheduler.max_steps == 300
            image = torch.randn(1, 4, 64, 64, 32)
            labels = torch.randint(0, 4, (1, 64, 64, 32))
            regions = torch.stack((labels > 0, labels > 1, labels > 2), 1)
            with torch.no_grad():
                out = trainer.network(image)
            targets = [torch.nn.functional.interpolate(regions.float(), size=z.shape[2:], mode='nearest').bool() for z in out]
            batch = {'data': image, 'target': targets}
            trainer.network.set_epoch(49)
            trainer.network.train()
            trainer.train_step(batch)
            assert trainer.network.semantic.beta_logit.grad is None
            assert all(p.grad is None for p in trainer.network.semantic.router.parameters())
            for expert in trainer.network.semantic.experts:
                assert expert.seg_layers[-1].weight.grad.abs().sum() > 0
            # checkpoint written after Epoch 49: next epoch is 50, saved prediction is warmup.
            trainer.current_epoch = 49
            warm_checkpoint = str(Path(tmp) / (cls.__name__ + '_warm.pth'))
            trainer.save_checkpoint(warm_checkpoint)
            trainer.network.set_epoch(50)
            trainer.load_checkpoint(warm_checkpoint)
            assert trainer.current_epoch == 50
            assert not bool(trainer.network.semantic.fusion_active.item())
            trainer.on_train_epoch_start()
            assert bool(trainer.network.semantic.fusion_active.item())
            values = trainer.train_step(batch)
            assert np.isfinite(values['loss'])
            assert trainer.network.semantic.beta_logit.grad.abs().sum() > 0
            router_grad = trainer.network.semantic.router.head.weight.grad
            assert router_grad is None if cls.uniform_fusion else router_grad.abs().sum() > 0
            trainer.network.eval()
            with torch.no_grad():
                parts = trainer.network(image, return_aux=True)
                assert len(parts['output']) == len(targets)
                assert parts['weights'].shape == (1, 3, 4)
                assert torch.allclose(parts['weights'].sum(-1), torch.ones(1, 3))
                if cls.uniform_fusion:
                    assert torch.equal(parts['weights'], torch.full_like(parts['weights'], .25))
                probe = semantic_probe(trainer.network, parts, targets[0])
                json.dumps(probe, allow_nan=False)
                if cls.uniform_fusion:
                    assert probe['counterfactuals']['uniform_expert_weights']['mean_abs_probability_change'] == 0
                normal = parts['output'][0].clone()
                trainer.set_deep_supervision_enabled(False)
                assert torch.equal(normal, trainer.network(image))
            checkpoint = str(Path(tmp) / (cls.__name__ + '.pth'))
            trainer.save_checkpoint(checkpoint)
            with torch.no_grad():
                trainer.network.semantic.beta_logit.add_(3)
            trainer.load_checkpoint(checkpoint)
            trainer.network.eval()
            with torch.no_grad():
                assert torch.equal(normal, trainer.network(image))
            trainer._write_module_record('MODULE_GRAD_FEATURE', {'modules': trainer._semantic_monitor.epoch_summary()})
            trainer._semantic_monitor.close()
            report['training_contracts'][cls.__name__] = {'warmup_expert_gradients': True, 'joint_router_gradient': not cls.uniform_fusion,
                'phase_boundary_resume': True, 'tensor_and_deep_supervision_outputs': True, 'checkpoint_output_exact': True,
                'finite_contribution_and_gradient_logs': True}
            del trainer, parts, normal, image, out, targets, batch
            gc.collect()

        # Full planned tensor geometry without allocating GPU activations.
        seed_experiment(0)
        meta_net = classes[2].build_network_architecture(manager, config, 4, 3, True).to('meta').eval()
        meta_net.semantic.fusion_active = torch.tensor(True)
        with torch.no_grad():
            meta_output = meta_net(torch.empty(2, 4, *config.patch_size, device='meta'))
        assert tuple(meta_output[0].shape) == (2, 3, *config.patch_size)
        report['full_patch_meta_shapes_only'] = [list(z.shape) for z in meta_output]
        del meta_net, meta_output

        if args.loader:
            os.environ['nnUNet_n_proc_DA'] = '0'
            fingerprints = []
            for cls in classes[1:]:
                cfg = copy.deepcopy(plans)
                cfg['continue_training'] = False
                trainer = cls(cfg, '3d_fullres', 0, dataset, device=torch.device('cpu'))
                trainer._set_batch_size_and_oversample()
                tr, va = trainer.get_dataloaders()
                batch = next(tr)
                signature = hashlib.sha256(batch['data'].contiguous().numpy().tobytes())
                for target in batch['target']:
                    assert (target[:, 2] <= target[:, 1]).all() and (target[:, 1] <= target[:, 0]).all()
                    signature.update(target.contiguous().numpy().tobytes())
                fingerprints.append({'trainer': cls.__name__, 'data_and_target_sha256': signature.hexdigest(),
                                     'shape': list(batch['data'].shape), 'keys': [str(k) for k in batch.get('keys', [])]})
                del tr, va, trainer, batch
                gc.collect()
            assert len({r['data_and_target_sha256'] for r in fingerprints}) == 1
            report['real_augmented_train_batch_same_seed'] = fingerprints

        if args.gpu:
            assert torch.cuda.is_available(), '--gpu requires an available CUDA GPU'
            cfg = copy.deepcopy(plans)
            cfg['continue_training'] = False
            trainer = classes[2](cfg, '3d_fullres', 0, dataset, device=torch.device('cuda'))
            trainer.initialize()
            trainer.network.set_epoch(50)
            trainer.network.train()
            torch.cuda.reset_peak_memory_stats()
            image = torch.randn(2, 4, *config.patch_size, device='cuda')
            with torch.no_grad(), torch.autocast('cuda'):
                sizes = [z.shape[2:] for z in trainer.network(image)]
            labels = torch.randint(0, 4, (2, *config.patch_size), device='cuda')
            region = torch.stack((labels > 0, labels > 1, labels > 2), 1).float()
            target = [torch.nn.functional.interpolate(region, size=s, mode='nearest') for s in sizes]
            trainer.train_step({'data': image, 'target': target})
            torch.cuda.synchronize()
            report['gpu'] = {'patch': config.patch_size, 'batch': 2,
                             'amp_full_patch_backward': True, 'peak_allocated_GiB': torch.cuda.max_memory_allocated() / 2**30}
        Path(args.output).write_text(json.dumps(report, indent=2) + '\n')
        print('SEMANTIC_CHECK_OK', args.output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--loader', action='store_true')
    parser.add_argument('--output', default='/root/nnUNet/documentation/brats2021_semantic_checks.json')
    run(parser.parse_args())
