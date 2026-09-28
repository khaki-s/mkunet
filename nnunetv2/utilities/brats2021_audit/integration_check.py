"""Real loader/augmentation and native sliding-window checks, without training.

Outputs and any checkpoint are temporary. Reads train-fold labels only when
--loader is selected; never calls run_training or optimizer.step.
"""
import argparse
import copy
import gc
import json
import os
import tempfile
from pathlib import Path

import torch


def main(args):
    os.environ['nnUNet_preprocessed'] = '/root/autodl-tmp/data/nnUNet_preprocessed'
    os.environ['nnUNet_raw'] = '/root/autodl-tmp/data/nnUNet_raw'
    os.environ['nnUNet_compile'] = 'False'
    os.environ['nnUNet_wandb_enabled'] = 'False'
    os.environ['nnUNet_n_proc_DA'] = str(args.workers)
    from nnunetv2.training.nnUNetTrainer.variants.brats2021_audit import nnUNetTrainer_BraTSAudit200 as registry
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    from nnunetv2.utilities.find_objects import recursive_find_trainer_class_by_name
    root = Path(os.environ['nnUNet_preprocessed']) / 'Dataset137_BraTS2021'
    plans = json.loads((root / 'nnUNetPlans.json').read_text())
    dataset = json.loads((root / 'dataset.json').read_text())
    classes = [getattr(registry, name) for name in dir(registry) if name.startswith('nnUNetTrainer_')]
    result = {'trainer_initialization': [], 'sliding_window': {}, 'loader_workers': args.workers}
    torch.set_num_threads(4)
    with tempfile.TemporaryDirectory(prefix='brats2021_integration_') as tmp:
        os.environ['nnUNet_results'] = tmp
        for cls in classes:
            assert recursive_find_trainer_class_by_name(cls.__name__) is cls
            cfg = copy.deepcopy(plans)
            cfg['continue_training'] = False  # this checkout's logger extension
            trainer = cls(cfg, '3d_fullres', 0, dataset, device=torch.device('cpu'))
            trainer.initialize()
            assert trainer.num_epochs == 300 and trainer.batch_size == 2
            assert trainer.initial_lr == .01 and trainer.oversample_foreground_percent == .33
            assert trainer.optimizer.param_groups[0]['lr'] == .01
            result['trainer_initialization'].append(cls.__name__)
            if args.loader and cls.__name__ == 'nnUNetTrainer_BraTSHardLoss300':
                train, val = trainer.get_dataloaders()
                try:
                    batch = next(train)
                    assert batch['data'].shape == (2, 4, 128, 160, 112)
                    assert len(batch['target']) == 5
                    for target in batch['target']:
                        assert target.shape[1] == 3
                        assert (target[:, 2] <= target[:, 1]).all() and (target[:, 1] <= target[:, 0]).all()
                    if 'keys' in batch:
                        split = json.loads((root / 'splits_final.json').read_text())[0]
                        assert set(batch['keys']).issubset(split['train'])
                    result['real_augmented_batch_shapes'] = [list(t.shape) for t in batch['target']]
                    del batch
                finally:
                    for loader in (train, val):
                        if hasattr(loader, '_finish'):
                            loader._finish()
            if args.sliding and trainer.mode in ('baseline', 'moe', 'frequency', 'mamba'):
                net = trainer.network.cuda().eval()
                net.decoder.deep_supervision = False
                predictor = nnUNetPredictor(use_mirroring=True, use_gaussian=True,
                                            device=torch.device('cuda'), allow_tqdm=False)
                predictor.manual_initialization(net, trainer.plans_manager, trainer.configuration_manager,
                                                None, dataset, cls.__name__, (0, 1, 2))
                # Larger than a real patch in two dimensions: overlapping tiles,
                # Gaussian weighting and every configured mirror axis are active.
                volume = torch.randn(4, 129, 160, 113)
                logits = predictor.predict_sliding_window_return_logits(volume)
                assert logits.shape == (3, 129, 160, 113) and torch.isfinite(logits).all()
                segmentation = trainer.label_manager.convert_logits_to_segmentation(logits)
                assert torch.isin(segmentation, torch.tensor([0, 1, 2, 3], device=segmentation.device)).all()
                result['sliding_window'][trainer.mode] = list(logits.shape)
                del predictor, logits, segmentation, volume, net
            del trainer
            gc.collect()
            torch.cuda.empty_cache()
    Path(args.output).write_text(json.dumps(result, indent=2) + '\n')
    print('PASS:', json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--loader', action='store_true')
    parser.add_argument('--sliding', action='store_true')
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--output', default='brats2021_integration.json')
    main(parser.parse_args())
