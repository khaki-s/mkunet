#!/usr/bin/env python3
"""Export frozen leave-one-modality-out predictions for the SAM3 study."""
import argparse
import fcntl
from pathlib import Path

import nibabel as nib
import torch

from predict_prompt_expert import initialize, publish_probabilities
from nnunetv2.inference.export_prediction import export_prediction_from_logits
from nnunetv2.utilities.brats2021_audit_v2.missing_modality import mask_from_code
from nnunetv2.utilities.brats2021_audit_v2.missing_predictor import MissingModalityPredictor
from nnunetv2.utilities.brats2021_audit_v2.interactive.foundation_prompt_v2.common import (
    MODALITIES, check_file, file_identity, read_json, read_labels, same_grid, write_json,
)
from nnunetv2.utilities.brats2021_audit_v2.interactive.foundation_prompt_v2.sam_protocol.cohorts import rows
from nnunetv2.utilities.brats2021_audit_v2.interactive.foundation_prompt_v2.sam_protocol.expert_modality import (
    MODALITY_CODES,
)

TRAINER_MODULE = ('nnunetv2.training.nnUNetTrainer.variants.brats2021_audit.'
                  'nnUNetTrainer_BraTSAgentGeneralistV2Loss300')


def predict(args):
    model = Path(args.model)
    checkpoint = args.checkpoint
    weights = {name: file_identity(model / name) for name in ('plans.json', 'dataset.json', 'fold_0/' + checkpoint)}
    output = Path(args.output)
    locks = output / '.locks'
    locks.mkdir(parents=True, exist_ok=True)
    source = {'schema': 'frozen_modality_expert_v1', 'weights': weights,
              'inference': 'nnUNet controlled preprocessing with four original inputs',
              'mirroring': True, 'gt_used': False}
    if args.save_probabilities:
        source['probabilities_exported'] = True
    with (locks / 'model').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        manifest_path = output / 'manifest.json'
        if manifest_path.exists():
            if read_json(manifest_path) != source:
                raise ValueError('Existing modality archive uses another checkpoint')
        else:
            write_json(manifest_path, source)
    selection = rows(args.cohort, args.split)[args.shard_index::args.num_shards]
    if args.reverse:
        selection = selection[::-1]
    if args.limit is not None:
        selection = selection[:args.limit]
    predictor = preprocessor = None
    torch.set_num_threads(3)
    torch.backends.cudnn.benchmark = False
    for row in selection:
        key = row['case_id']
        if row['mask_code'] != 15 or 'gt' in row or set(row['modalities']) != set(MODALITIES):
            raise ValueError('Leave-one-out cues need the four MRI inputs and a label-free row')
        with (locks / key).open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            inputs = [row['modalities'][name] for name in MODALITIES]
            for identity in [*inputs, row['prediction']]:
                check_file(identity)
            reference = nib.load(row['prediction']['path'])
            image_data = properties = None
            for code in args.codes:
                folder = output / 'predictions' / f'mask_{code:02d}'
                folder.mkdir(parents=True, exist_ok=True)
                binding = {'row': row, 'weights': weights, 'mask_code': code,
                           'controlled_preprocessing': True, 'mirroring': True}
                if args.save_probabilities:
                    binding['probabilities_exported'] = True
                receipt = folder / f'{key}.json'
                if receipt.exists():
                    previous = read_json(receipt)
                    if previous['binding'] != binding or previous['gt_used'] is not False:
                        raise ValueError(f'{key}: model or MRI inputs changed')
                    check_file(previous['mask'])
                    if args.save_probabilities:
                        check_file(previous['probabilities'])
                    continue
                if predictor is None:
                    predictor = MissingModalityPredictor(device=torch.device(args.device),
                                                         perform_everything_on_device=False, use_mirroring=True,
                                                         verbose=False, allow_tqdm=False)
                    initialize(predictor, model, checkpoint, TRAINER_MODULE)
                    preprocessor = predictor.configuration_manager.preprocessor_class(verbose=False)
                if image_data is None:
                    data, _, properties = preprocessor.run_case([item['path'] for item in inputs], None,
                                                                 predictor.plans_manager,
                                                                 predictor.configuration_manager,
                                                                 predictor.dataset_json)
                    image_data = torch.from_numpy(data)
                logits, _ = predictor.predict_with_mask(image_data, mask_from_code(code), case_id=key,
                                                        collect_routes=False)
                temporary = folder / f'{key}.partial'
                export_prediction_from_logits(logits, properties, predictor.configuration_manager,
                                              predictor.plans_manager, predictor.dataset_json, str(temporary),
                                              save_probabilities=args.save_probabilities, num_threads_torch=3)
                staged = temporary.with_suffix('.partial.nii.gz')
                image = nib.load(staged)
                same_grid(image, reference)
                label = read_labels(image)
                path = folder / f'{key}.nii.gz'
                staged.replace(path)
                record = {'binding': binding, 'mask': file_identity(path), 'gt_used': False,
                          'native_expert_patch': list(predictor.configuration_manager.patch_size)}
                if args.save_probabilities:
                    record['probabilities'] = publish_probabilities(temporary, folder, key)
                write_json(receipt, record)
                print(f'{key} mask_{code:02d}: {(label > 0).sum()} WT voxels', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('cohort', 'model', 'output'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--split', choices=('train', 'validation'), required=True)
    parser.add_argument('--codes', nargs='+', type=int, choices=(*MODALITY_CODES, 15), default=MODALITY_CODES)
    parser.add_argument('--shard-index', type=int, default=0)
    parser.add_argument('--num-shards', type=int, default=1)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--reverse', action='store_true')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--checkpoint', choices=('checkpoint_best.pth', 'checkpoint_final.pth'),
                        default='checkpoint_final.pth')
    parser.add_argument('--save-probabilities', action='store_true')
    args = parser.parse_args()
    if not 0 <= args.shard_index < args.num_shards or args.limit is not None and args.limit < 1:
        parser.error('Invalid prediction partition')
    predict(args)
