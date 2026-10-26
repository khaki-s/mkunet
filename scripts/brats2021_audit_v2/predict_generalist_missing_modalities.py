#!/usr/bin/env python
"""Predict from a JSON list of named, physically available NIfTI files."""
import argparse
from pathlib import Path
import torch

from nnunetv2.utilities.brats2021_audit_v2.missing_predictor import load_predictor, read_manifest
from nnunetv2.utilities.brats2021_audit_v2.agent_evaluate import aggregate_case_routes, file_sha256, write_csv, write_json
from nnunetv2.utilities.brats2021_audit_v2.missing_modality import mask_codes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--model-folder', type=Path, required=True)
    parser.add_argument('--checkpoint', default='checkpoint_final.pth')
    parser.add_argument('--fold', default='0')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--disable-tta', action='store_true')
    parser.add_argument('--save-probabilities', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(4)
    cases = read_manifest(args.manifest)
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError('Choose an empty prediction directory')
    args.output.mkdir(parents=True, exist_ok=True)
    predictor = load_predictor(args.model_folder, args.checkpoint, folds=(args.fold,),
                              device=torch.device(args.device), mirroring=not args.disable_tta)
    record = {'status': 'running', 'protocol': 'available_files', 'manifest_sha256': file_sha256(args.manifest),
              'model_folder': str(args.model_folder.resolve()), 'checkpoint': args.checkpoint, 'fold': args.fold,
              'tta': not args.disable_tta, 'cases': cases}
    model_base = args.model_folder.parent if args.model_folder.name.startswith('fold_') else args.model_folder
    fold_dir = args.model_folder if args.model_folder.name.startswith('fold_') else model_base / ('fold_' + args.fold)
    record['checkpoint_sha256'] = file_sha256(fold_dir / args.checkpoint)
    write_json(args.output / 'manifest.json', record)
    all_routes = []
    for case in cases:
        routes, properties = predictor.predict_available_case(case, args.output / case['case_id'], save_probabilities=args.save_probabilities)
        code = int(mask_codes(torch.tensor(properties['modality_mask'])[None])[0])
        all_routes.extend(aggregate_case_routes(routes, case['case_id'], code))
        write_json(args.output / (case['case_id'] + '.routing.json'), {'mask_code': code, 'routes': routes})
        print(f'Predicted {case["case_id"]}, mask={code}', flush=True)
    if all_routes:
        write_csv(args.output / 'routing_per_case.csv', all_routes)
    record['status'] = 'complete'
    write_json(args.output / 'manifest.json', record)


if __name__ == '__main__':
    main()
