"""Run a frozen four-modality segmentation expert for independent prompt evidence."""
import argparse
import fcntl
from importlib import import_module
from pathlib import Path
from time import monotonic
from unittest.mock import patch

import nibabel as nib
import numpy as np
import torch

from nnunetv2.inference import predict_from_raw_data
from nnunetv2.inference.export_prediction import export_prediction_from_logits
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.utilities.brats2021_audit_v2.interactive.foundation_prompt_v2.common import (
    MODALITIES, check_file, file_identity, read_json, read_labels, same_grid, write_json,
)
from nnunetv2.utilities.brats2021_audit_v2.interactive.foundation_prompt_v2.sam_protocol.cohorts import rows


def initialize(predictor, model, checkpoint, trainer_module):
    if not trainer_module:
        return predictor.initialize_from_trained_model_folder(str(model), ('0',), checkpoint)
    module = import_module(trainer_module)
    with patch.object(predict_from_raw_data, 'recursive_find_trainer_class_by_name', lambda name: getattr(module, name)):
        return predictor.initialize_from_trained_model_folder(str(model), ('0',), checkpoint)


def publish_probabilities(temporary, output, key):
    path = output / f'{key}.npz'
    temporary.with_suffix('.partial.npz').replace(path)
    temporary.with_suffix('.partial.pkl').unlink(missing_ok=True)
    return file_identity(path)


def predict(args):
    selected = rows(args.cohort, args.split)[args.shard_index::args.num_shards]
    if args.reverse:
        selected = selected[::-1]
    if args.limit is not None:
        selected = selected[:args.limit]
    model = Path(args.model)
    checkpoint = args.checkpoint
    source = {name: file_identity(model / name) for name in ('plans.json', 'dataset.json', 'fold_0/' + checkpoint)}
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    locks = output / '.locks'
    locks.mkdir(exist_ok=True)
    predictor = None
    torch.backends.cudnn.benchmark = False
    torch.set_num_threads(3)
    for row in selected:
        start = monotonic()
        key = row['case_id']
        with (locks / key).open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if not all(row['availability']) or set(row['modalities']) != set(MODALITIES) or 'gt' in row:
                raise ValueError('This expert requires the four original MRI inputs and a label-free record')
            binding = {'row': row, 'model': source, 'mirroring': True}
            if args.save_probabilities:
                binding['probabilities_exported'] = True
            receipt = output / f'{key}.json'
            if receipt.is_file():
                cached = read_json(receipt)
                if cached['binding'] != binding or cached['gt_used']:
                    raise ValueError(f'{key}: prediction was generated from another MRI or model')
                check_file(cached['mask'])
                if args.save_probabilities:
                    check_file(cached['probabilities'])
                continue
            inputs = [row['modalities'][name] for name in MODALITIES]
            for identity in (row['prediction'], *inputs):
                check_file(identity)
            if predictor is None:
                device = torch.device(args.device)
                predictor = nnUNetPredictor(device=device, perform_everything_on_device=False,
                                            use_mirroring=True, verbose=False, allow_tqdm=False)
                initialize(predictor, model, checkpoint, args.trainer_module)
                if predictor.configuration_manager.previous_stage_name is not None:
                    raise ValueError('A cascaded expert would require a separately bound input mask')
                preprocessor = predictor.configuration_manager.preprocessor_class(verbose=False)
            data, _, properties = preprocessor.run_case([item['path'] for item in inputs], None,
                                                        predictor.plans_manager, predictor.configuration_manager,
                                                        predictor.dataset_json)
            logits = predictor.predict_logits_from_preprocessed_data(torch.from_numpy(data)).cpu()
            temporary = output / f'{key}.partial'
            export_prediction_from_logits(logits, properties, predictor.configuration_manager,
                                          predictor.plans_manager, predictor.dataset_json, str(temporary),
                                          save_probabilities=args.save_probabilities, num_threads_torch=3)
            staged = temporary.with_suffix('.partial.nii.gz')
            result = nib.load(staged)
            same_grid(result, nib.load(row['prediction']['path']))
            labels = read_labels(result)
            path = output / f'{key}.nii.gz'
            staged.replace(path)
            record = {'binding': binding, 'mask': file_identity(path), 'gt_used': False,
                      'native_expert_patch': list(predictor.configuration_manager.patch_size),
                      'seconds': monotonic() - start}
            if args.save_probabilities:
                record['probabilities'] = publish_probabilities(temporary, output, key)
            write_json(receipt, record)
            print(f'{model.name} {key}: {(labels > 0).sum()} WT voxels in {monotonic() - start:.1f}s', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('cohort', 'model', 'output'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--split', choices=('train', 'validation'), default='train')
    parser.add_argument('--shard-index', type=int, default=0)
    parser.add_argument('--num-shards', type=int, default=1)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--reverse', action='store_true')
    parser.add_argument('--trainer-module', help='Exact local trainer module for a known checkpoint')
    parser.add_argument('--checkpoint', choices=('checkpoint_best.pth', 'checkpoint_final.pth'),
                        default='checkpoint_final.pth')
    parser.add_argument('--save-probabilities', action='store_true')
    args = parser.parse_args()
    if not 0 <= args.shard_index < args.num_shards or (args.limit is not None and args.limit < 1):
        parser.error('Invalid prediction partition')
    predict(args)


if __name__ == '__main__':
    main()
