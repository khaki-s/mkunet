#!/usr/bin/env python
"""Evaluate one checkpoint on 15 subsets: preprocessed val or available files."""
import argparse
import json
import os
from pathlib import Path
import torch

from nnunetv2.training.dataloading.nnunet_dataset import infer_dataset_class
from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name
from nnunetv2.utilities.brats2021_audit.agent_evaluate import evaluate_cases, evaluate_available, file_sha256, paired_comparison, preprocessed_input_fingerprints
from nnunetv2.utilities.brats2021_audit.missing_predictor import load_predictor, read_manifest
from nnunetv2.utilities.brats2021_audit.missing_modality import MASK_CODES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', default='137')
    parser.add_argument('--trainer', default='nnUNetTrainer_BraTSSemanticMissing300')
    parser.add_argument('--plans', default='nnUNetPlans')
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--split', choices=('val', 'test'), default='val')
    parser.add_argument('--model-folder', type=Path)
    parser.add_argument('--checkpoint', default='checkpoint_final.pth')
    parser.add_argument('--masks', default='all15', help='all15 or comma-separated integer codes')
    parser.add_argument('--manifest', type=Path, help='Named source files; preprocess each subset independently')
    parser.add_argument('--gt', type=Path, help='Internal-label reference NIfTI folder (required with manifest)')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--baseline-csv', type=Path)
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    parser.add_argument('--disable-tta', action='store_true')
    parser.add_argument('--save-probabilities', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(4)
    codes = MASK_CODES if args.masks == 'all15' else tuple(int(c) for c in args.masks.split(','))
    name = maybe_convert_to_dataset_name(int(args.dataset) if args.dataset.isdigit() else args.dataset)
    model = args.model_folder or Path(os.environ['nnUNet_results']) / name / f'{args.trainer}__{args.plans}__3d_fullres'
    predictor = load_predictor(model, args.checkpoint, folds=(args.fold,), device=torch.device(args.device), mirroring=not args.disable_tta)
    fold_dir = model if model.name.startswith('fold_') else model / f'fold_{args.fold}'
    provenance = {'checkpoint': str((fold_dir / args.checkpoint).resolve()), 'checkpoint_sha256': file_sha256(fold_dir / args.checkpoint),
                  'split': args.split, 'fold': args.fold}
    root = Path(os.environ['nnUNet_preprocessed']) / name
    split = json.loads((root / 'splits_final.json').read_text())[args.fold]
    if args.manifest:
        if args.gt is None:
            parser.error('--gt is required with --manifest')
        cases = read_manifest(args.manifest)
        keys = {c['case_id'] for c in cases}
        if args.split == 'val' and keys != set(split['val']):
            raise ValueError('Validation manifest must match all validation IDs')
        if args.split == 'test' and keys.intersection(split['train'] + split['val']):
            raise ValueError('Test manifest overlaps train/val')
        evaluate_available(predictor, cases, args.gt, args.output, codes=codes,
                           save_probabilities=args.save_probabilities, provenance=provenance)
    else:
        if args.split != 'val':
            parser.error('Test evaluation requires --manifest and --gt')
        folder = root / predictor.configuration_manager.data_identifier
        dataset = infer_dataset_class(str(folder))(str(folder), list(split['val']))
        cache = {}
        def source(key, code):
            if cache.get('key') != key:
                data, _, _, properties = dataset.load_case(key)
                cache.update(key=key, data=torch.from_numpy(data[:]), properties=properties)
            return cache['data'], cache['properties']
        provenance.update(data_folder=str(folder.resolve()), splits_sha256=file_sha256(root / 'splits_final.json'),
                          input_sha256=preprocessed_input_fingerprints(folder, split['val']))
        evaluate_cases(predictor, list(split['val']), source, args.gt or root / 'gt_segmentations', args.output,
                       codes=codes, save_probabilities=args.save_probabilities, provenance=provenance)
    if args.baseline_csv:
        paired_comparison(args.output / 'per_case_metrics.csv', args.baseline_csv, args.output / 'paired_comparison.json')


if __name__ == '__main__':
    main()
