"""Independent dense-mask refinement with frozen nnInteractive and Atlas fusion.

This is one automatic dense refinement, not a point-budget experiment. The
upstream FusionAdapter/loss are used in train320; validation and deployment never
update model weights. Query ground truth is read only by collect's supervised
pair export and the separate scorer, never by the foundation model or predictor.
"""
from __future__ import annotations

import argparse
import csv
import gc
import importlib.metadata
import json
from pathlib import Path
import time

import nibabel as nib
import numpy as np
import torch

from . import atlas_adapter as adapter
from .common import (manifest, load_case, read_json, write_json, file_identity,
                     check_file, model_identity, save_mask, region_metrics, REGIONS)
from ..evaluation import read_labels, same_grid
from ..inference_runtime import new_session, prediction_context, NATIVE_PATCH_SIZE
from .parallel import partition, add_partition_arguments, merge_collections

MODEL = '/data02/chujingyi/model/nnInteractive/nnInteractive_v1.0'
HEADS = ('WT', 'TC', 'ET')
CHANNELS = {'WT': 3, 'TC': 1, 'ET': 1}


def contract(source, model_folder):
    check_file(source['initial_checkpoint'])
    folder = Path(__file__).parent
    files = [folder / p for p in ('atlas_pipeline.py', 'atlas_adapter.py', 'common.py')]
    files += [folder.parent / p for p in ('evaluation_manifest.py', 'interaction_metrics.py')]
    files += [folder.parent / 'inference_runtime.py']
    return {'method': 'atlas_dense_train320_v1', 'initial_checkpoint_sha256': source['initial_checkpoint']['sha256'],
            'split_sha256': source['splits_file']['sha256'], 'nninteractive': model_identity(model_folder),
            'sources': {p.name: file_identity(p)['sha256'] for p in files},
            'atlas_source_sha256': file_identity(adapter.DEFAULT_SOURCE)['sha256'],
            'nninteractive_version': importlib.metadata.version('nnInteractive'),
            'modality_by_head': CHANNELS, 'foundation_frozen': True,
            'nninteractive_patch_size': list(NATIVE_PATCH_SIZE), 'patch_policy': 'checkpoint_native_only',
            'cudnn_benchmark': False,
            'prompt': 'current_binary_dense_mask_only', 'empty_initial_head': 'keep_empty',
            'hierarchy': 'WT=max(WT,TC,ET);TC=max(TC,ET);threshold>=0.5',
            'budget': 'one dense refinement; at most three nnInteractive prompt calls'}


class DenseRefiner:
    def __init__(self, model_folder, device):
        self.session = new_session(model_folder, device)

    @torch.inference_mode()
    def predict_head(self, mri, current, head):
        initial = adapter._region(current, head).astype(np.uint8)
        if not initial.any():
            return initial, 0
        self.session.set_image(np.ascontiguousarray(mri[CHANNELS[head]:CHANNELS[head]+1], dtype=np.float32))
        output = torch.zeros(current.shape, dtype=torch.uint8, device='cpu')
        self.session.set_target_buffer(output)
        self.session.reset_interactions()
        with prediction_context(self.session):
            self.session.add_initial_seg_interaction(initial, run_prediction=True)
        return output.numpy().copy(), 1

    def close(self):
        executor = getattr(self.session, 'executor', None)
        if executor is not None:
            executor.shutdown(wait=True)
        del self.session


def _release(engine):
    if hasattr(engine, 'close'):
        engine.close()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _dense(mri, current, model_folder, device, factory):
    """No GT inputs. Recreate/release session per case to bound retained memory."""
    engine = factory(model_folder, device)
    outputs, calls = {}, 0
    try:
        for head in HEADS:
            result, count = engine.predict_head(mri, current, head)
            result = np.asarray(result)
            if result.shape != current.shape or not np.isin(result, (0, 1)).all():
                raise ValueError('Foundation model returned an invalid binary mask')
            outputs[head] = result.astype(np.uint8)
            calls += count
    finally:
        _release(engine)
    return outputs, calls


def _resume(path, binding):
    if Path(path).exists():
        prior = read_json(path)
        if prior['binding'] != binding:
            raise ValueError('Output already has a different protocol/input/model; use a new directory')
        for entry in prior['completed'].values():
            for identity in entry['files']:
                check_file(identity)
        return prior['completed']
    return {}


def _write_progress(path, binding, completed, total):
    write_json(path, {'binding': binding, 'completed': completed,
                      'full_cohort': len(completed) == total})


def _write_pairs(root, index, manifest_path, method, completed):
    write_json(root / 'adapter_pairs.json', {'schema': 'foundation_adapter_pairs_v1', 'split': 'train',
               'source_manifest': str(Path(manifest_path).resolve()), 'dense_contract': method,
               'collection_identity': file_identity(index),
               'cases': [r for c in completed.values() for r in c['pairs']]})


def _check_modalities(row):
    if row.get('availability', [1]*4) != [1, 1, 1, 1]:
        raise ValueError('This dense pipeline is fixed to the four-present-modality experiment')


def collect(args, *, refiner_factory=DenseRefiner):
    source = manifest(args.manifest, allowed=('train',))
    method = contract(source, args.model_folder)
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    binding = {'schema': 'atlas_dense_collect_v1', 'source': file_identity(args.manifest),
               'contract': method, 'limit_cases': args.limit_cases}
    rows = source['cases'][:args.limit_cases] if args.limit_cases else source['cases']
    rows = partition(rows, args, binding)
    index = root / 'collection.json'
    completed = _resume(index, binding)
    for row in rows:
        key = row['case_id']
        if key in completed:
            continue
        _check_modalities(row)
        started = time.monotonic()
        # Deliberately load no labels before completing all foundation inference.
        mri, current, reference = load_case({k:v for k,v in row.items() if k != 'gt'})
        dense, calls = _dense(mri, current, args.model_folder, args.device, refiner_factory)
        dense_seconds = time.monotonic() - started
        check_file(row['gt'])
        truth = nib.load(row['gt']['path'])
        same_grid(truth, reference)
        gt = read_labels(truth)
        case_dir = root / key
        case_dir.mkdir(parents=True, exist_ok=True)
        pairs, files = [], []
        for i, head in enumerate(HEADS):
            dense_file = case_dir / f'dense_{head}.nii.gz'
            files.append(save_mask(dense_file, dense[head], reference))
            pair_file = case_dir / f'pair_{head}.npz'
            # The proposal encodes this head only; training extracts that head.
            np.savez_compressed(pair_file, mri=mri, baseline=current, proposal=dense[head]*(i+1),
                                gt=gt, head=np.array(i, dtype=np.int64))
            identity = file_identity(pair_file)
            files.append(identity)
            pairs.append({'case_id':key, 'file':str(pair_file.resolve()), 'sha256':identity['sha256']})
        completed[key] = {'files': files, 'pairs': pairs, 'nninteractive_calls': calls,
                          'dense_seconds_including_load': dense_seconds,
                          'case_seconds_including_export': time.monotonic()-started}
        _write_progress(index, binding, completed, len(source['cases']))
        _write_pairs(root, index, args.manifest, method, completed)
        print(f'collect {len(completed)}/{len(rows)} {key}, calls={calls}', flush=True)
        del mri, current, gt, dense
        gc.collect()
    if not index.exists():
        _write_progress(index, binding, completed, len(source['cases']))
    # A crash can happen after committing collection.json but before publishing
    # adapter_pairs.json. Rebuild it even when resume skipped every case.
    _write_pairs(root, index, args.manifest, method, completed)
    return root / 'adapter_pairs.json'


def train(args):
    pairs = read_json(args.pairs)
    if pairs.get('dense_contract', {}).get('method') != 'atlas_dense_train320_v1':
        raise ValueError('Expected dense-pipeline pairs; do not silently mix point-refinement pairs')
    source = manifest(pairs['source_manifest'], allowed=('train',))
    if pairs['dense_contract'] != contract(source, args.model_folder):
        raise ValueError('Foundation/source protocol changed since collection')
    check_file(pairs['collection_identity'])
    for row in pairs['cases']:
        check_file({'path':row['file'], 'sha256':row['sha256']})
    path = adapter.fit_adapter(args.pairs, args.output, epochs=args.epochs, device=args.device,
                               patch_size=args.patch_size, steps_per_epoch=args.steps_per_epoch,
                               base_ch=args.base_ch, splits_path=source['splits_file']['path'],
                               heads=getattr(args, 'heads', HEADS),
                               sampling=getattr(args, 'sampling', 'disagreement'),
                               cache_pairs=getattr(args, 'cache_pairs', 0))
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    checkpoint['metadata']['cohort_contract'] = pairs['dense_contract']
    torch.save(checkpoint, path)
    write_json(Path(args.output)/'metadata.json', checkpoint['metadata'])
    return Path(path)


def project_probabilities(probabilities):
    p = np.stack(probabilities).astype(np.float32)
    if p.ndim != 4 or not np.isfinite(p).all() or p.min() < 0 or p.max() > 1:
        raise ValueError('Expected finite WT/TC/ET probabilities in [0,1]')
    p[1] = np.maximum(p[1], p[2])
    p[0] = np.maximum(p[0], p[1])
    labels = np.zeros(p.shape[1:], np.uint8)
    labels[p[0] >= .5] = 1
    labels[p[1] >= .5] = 2
    labels[p[2] >= .5] = 3
    return labels


def _check_lock(path, method, checkpoint):
    if not path:
        raise ValueError('Test prediction requires --validation-lock from a passed full val80 evaluation')
    locked = read_json(path)
    if locked.get('schema') != 'atlas_dense_val80_lock_v1' or not locked.get('passed') or locked.get('case_count') != 80:
        raise ValueError('Expected a passed val80 lock')
    check_file(locked['score'])
    check_file(locked['checkpoint'])
    if locked['contract'] != method or locked['checkpoint'] != file_identity(checkpoint):
        raise ValueError('Test method/checkpoint differs from validated frozen protocol')


def predict(args, *, refiner_factory=DenseRefiner):
    source = manifest(args.manifest, deployment=True, allowed=('validation','test'))
    method = contract(source, args.model_folder)
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    if state['metadata'].get('cohort_contract') != method or set(state['models']) != {'0','1','2'}:
        raise ValueError('Need all three trained heads with this exact dense-refinement contract')
    if source['split'] == 'test':
        _check_lock(args.validation_lock, method, args.checkpoint)
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    binding = {'schema':'atlas_dense_predictions_v1', 'source':file_identity(args.manifest),
               'contract':method, 'checkpoint':file_identity(args.checkpoint),
               'split':source['split'], 'limit_cases':args.limit_cases,
               'validation_lock':file_identity(args.validation_lock) if args.validation_lock else None}
    index = root / 'predictions.json'
    completed = _resume(index, binding)
    rows = source['cases'][:args.limit_cases] if args.limit_cases else source['cases']
    for row in rows:
        key = row['case_id']
        if key in completed:
            continue
        _check_modalities(row)
        started = time.monotonic()
        mri, current, reference = load_case(row)
        dense, calls = _dense(mri, current, args.model_folder, args.device, refiner_factory)
        dense_seconds = time.monotonic() - started
        probabilities = []
        for i, head in enumerate(HEADS):
            result = adapter.fuse_adapter(args.checkpoint, current, dense[head]*(i+1), mri, head, device=args.device)
            probabilities.append(result['probability'])
        final = project_probabilities(probabilities)
        case_dir = root / key
        case_dir.mkdir(parents=True, exist_ok=True)
        baseline_file = save_mask(case_dir/'baseline.nii.gz', current, reference)
        final_file = save_mask(case_dir/'final.nii.gz', final, reference)
        files = [baseline_file, final_file] + [save_mask(case_dir/f'dense_{h}.nii.gz', dense[h], reference) for h in HEADS]
        completed[key] = {'files':files, 'baseline':baseline_file, 'final':final_file,
                          'nninteractive_calls':calls, 'dense_seconds_including_load':dense_seconds,
                          'total_seconds_including_export':time.monotonic()-started,
                          'gt_used':False, 'changed_voxels':int(np.count_nonzero(final != current))}
        _write_progress(index, binding, completed, len(source['cases']))
        print(f'predict {len(completed)}/{len(rows)} {key}, calls={calls}', flush=True)
        del mri, current, dense, probabilities, final, result
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return index


def score(args):
    predictions = read_json(args.predictions)
    binding = predictions['binding']
    if binding.get('schema') != 'atlas_dense_predictions_v1':
        raise ValueError('Wrong prediction format')
    check_file(binding['source'])
    check_file(binding['checkpoint'])
    deploy = manifest(binding['source']['path'], deployment=True, allowed=('validation','test'))
    truth = manifest(args.manifest, allowed=(deploy['split'],))
    if truth['splits_file'] != deploy['splits_file'] or truth['initial_checkpoint'] != deploy['initial_checkpoint']:
        raise ValueError('Scoring source/model/split mismatch')
    deployment_rows = {r['case_id']:r for r in deploy['cases']}
    truth_rows = {r['case_id']:r for r in truth['cases']}
    if set(truth_rows) != set(deployment_rows) or not set(predictions['completed']) <= set(truth_rows):
        raise ValueError('Scoring patient mismatch')
    rows = []
    for key, result in predictions['completed'].items():
        for identity in result['files']:
            check_file(identity)
        row = truth_rows[key]
        if {k:v for k,v in row.items() if k != 'gt'} != deployment_rows[key]:
            raise ValueError('Scoring MRI/availability/initial prediction differs from deployment inputs')
        check_file(row['gt']); check_file(row['prediction'])
        gt_ref, before_ref, after_ref = (nib.load(x['path']) for x in (row['gt'], result['baseline'], result['final']))
        initial_ref = nib.load(row['prediction']['path'])
        for image in (before_ref, after_ref, initial_ref):
            same_grid(gt_ref, image)
        gt, before, after = map(read_labels, (gt_ref, before_ref, after_ref))
        if not np.array_equal(before, read_labels(initial_ref)):
            raise ValueError('Saved baseline differs from the original prediction')
        old, new = region_metrics(before, gt), region_metrics(after, gt)
        for region in REGIONS:
            b, a = old[region]['dice'], new[region]['dice']
            selector = (lambda x:x == 1) if region == 'ED' else ((lambda x:x == 2) if region == 'NCR'
                        else (lambda x:adapter._region(x,region)))
            added_empty_fp = int(np.count_nonzero(selector(after) & ~selector(before))) if old[region]['n_ref']==0 else 0
            rows.append({'case_id':key, 'region':region,
                         'baseline_pct':None if b is None else 100*b,
                         'final_pct':None if a is None else 100*a,
                         'gain_pp':None if b is None or a is None else 100*(a-b),
                         'baseline_both_empty':b is None, 'final_both_empty':a is None,
                         'gt_empty':old[region]['n_ref']==0, 'empty_gt_new_fp_voxels':added_empty_fp,
                         'secondary_empty_as_one_gain_pp':100*((1. if a is None else a)-(1. if b is None else b))})
    if not rows:
        raise ValueError('No completed cases to score')
    summary = {}
    for region in REGIONS:
        subset = [r for r in rows if r['region'] == region]
        paired = [r for r in subset if r['gain_pp'] is not None]
        gains = np.array([r['gain_pp'] for r in paired])
        secondary = np.array([r['secondary_empty_as_one_gain_pp'] for r in subset])
        summary[region] = {'n':len(paired), 'total_cases':len(subset),
                           'baseline_pct':float(np.mean([r['baseline_pct'] for r in paired])) if paired else None,
                           'final_pct':float(np.mean([r['final_pct'] for r in paired])) if paired else None,
                           'mean_gain_pp':float(gains.mean()) if paired else None,
                           'improved_fraction':float((gains > 1e-4).mean()) if paired else None,
                           'worsened_fraction':float((gains < -1e-4).mean()) if paired else None,
                           'secondary_empty_as_one_mean_gain_pp':float(secondary.mean()),
                           'secondary_empty_as_one_improved_fraction':float((secondary > 1e-4).mean()),
                           'secondary_empty_as_one_worsened_fraction':float((secondary < -1e-4).mean()),
                           'empty_gt_new_fp_voxels':sum(r['empty_gt_new_fp_voxels'] for r in subset),
                           'empty_gt_new_fp_case_count':sum(r['empty_gt_new_fp_voxels'] > 0 for r in subset)}
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    with (root/'per_case.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader()
        for row in rows:
            writer.writerow({k:(f'{v:.3f}' if isinstance(v,float) else v) for k,v in row.items()})
    wide = []
    for key in predictions['completed']:
        entry = {'case_id':key}
        for row in rows:
            if row['case_id'] == key:
                for metric in ('baseline_pct','final_pct','gain_pp'):
                    entry[f"{row['region']}_{metric}"] = '' if row[metric] is None else f"{row[metric]:.3f}"
        wide.append(entry)
    with (root/'per_case_readable_percent.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(wide[0]));writer.writeheader();writer.writerows(wide)
    output = {'schema':'atlas_dense_score_v1', 'predictions':file_identity(args.predictions),
              'truth_manifest':file_identity(args.manifest), 'contract':binding['contract'],
              'checkpoint':binding['checkpoint'], 'split':deploy['split'],
              'full_cohort':set(predictions['completed']) == set(deployment_rows),
              'case_count':len(predictions['completed']), 'regions':summary, 'per_case':rows,
              'both_empty_policy':'primary: None; means use pairs with both Dice defined. Secondary: both-empty=1, all cases.',
              'mean_seconds':float(np.mean([r['total_seconds_including_export'] for r in predictions['completed'].values()])),
              'total_nninteractive_calls':sum(r['nninteractive_calls'] for r in predictions['completed'].values()),
              'budget':'one dense refinement; no point/box annotation counts'}
    write_json(root/'score.json', output)
    print(json.dumps(output['regions'],indent=2), flush=True)
    return root/'score.json'


def lock(args, *, expected_cases=80):
    result = read_json(args.score)
    if result.get('schema') != 'atlas_dense_score_v1' or result['split'] != 'validation' or not result['full_cohort'] or result['case_count'] != expected_cases:
        raise ValueError('Lock requires a completed independent val80 evaluation')
    for field in ('predictions','truth_manifest','checkpoint'):
        check_file(result[field])
    gains = {r:result['regions'][r]['mean_gain_pp'] for r in REGIONS}
    secondary = {r:result['regions'][r]['secondary_empty_as_one_mean_gain_pp'] for r in REGIONS}
    new_fp = sum(result['regions'][r]['empty_gt_new_fp_voxels'] for r in REGIONS)
    criterion = getattr(args, 'criterion', 'wt-primary')
    if criterion not in ('wt-primary', 'all-three'):
        raise ValueError('Unknown validation criterion')
    required = ('WT',) if criterion == 'wt-primary' else HEADS
    passed = all(gains[r] is not None and gains[r] > 1e-6 for r in required)
    if not passed:
        raise ValueError(f'Validation failed ({criterion}): gains={gains}. Do not run test.')
    if Path(args.output).exists():
        raise FileExistsError(args.output)
    write_json(args.output, {'schema':'atlas_dense_val80_lock_v1', 'passed':True,
               'case_count':result['case_count'], 'score':file_identity(args.score),
               'checkpoint':result['checkpoint'], 'contract':result['contract'],
               'criterion':criterion,
               'criteria':'mean WT Dice gain > 0; TC/ET improvements are secondary goals' if criterion == 'wt-primary'
                          else 'mean WT,TC,ET Dice gains all > 0',
               'all_three_improved':all(gains[r] is not None and gains[r] > 1e-6 for r in HEADS),
               'observed_gain_pp':gains, 'secondary_empty_as_one_gain_pp':secondary,
               'empty_gt_new_fp_voxels':new_fp})
    return Path(args.output)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest='command',required=True)
    for name in ('collect','predict'):
        sub=commands.add_parser(name)
        sub.add_argument('--manifest',required=True);sub.add_argument('--output',required=True)
        sub.add_argument('--model-folder',default=MODEL);sub.add_argument('--device',default='cuda:0')
        sub.add_argument('--limit-cases',type=int)
        if name == 'collect':
            add_partition_arguments(sub)
        if name=='predict':
            sub.add_argument('--checkpoint',required=True);sub.add_argument('--validation-lock')
    sub=commands.add_parser('train');sub.add_argument('--pairs',required=True);sub.add_argument('--output',required=True)
    sub.add_argument('--epochs',type=int,default=30);sub.add_argument('--steps-per-epoch',type=int)
    sub.add_argument('--patch-size',type=int,nargs=3,default=(64,64,64));sub.add_argument('--base-ch',type=int,default=16)
    sub.add_argument('--device',default='cuda:0');sub.add_argument('--model-folder',default=MODEL)
    sub.add_argument('--heads', choices=HEADS, nargs='+', default=HEADS)
    sub.add_argument('--sampling', choices=('disagreement', 'legacy'), default='disagreement')
    sub.add_argument('--cache-pairs', type=int, default=0, help='Bounded CPU feature cache per head (0 disables)')
    sub=commands.add_parser('merge-collections')
    sub.add_argument('--collections', nargs='+', required=True);sub.add_argument('--output', required=True)
    sub=commands.add_parser('merge-adapters')
    sub.add_argument('--checkpoints', nargs='+', required=True);sub.add_argument('--output', required=True)
    sub=commands.add_parser('score');sub.add_argument('--predictions',required=True);sub.add_argument('--manifest',required=True);sub.add_argument('--output',required=True)
    sub=commands.add_parser('lock');sub.add_argument('--score',required=True);sub.add_argument('--output',required=True)
    sub.add_argument('--criterion',choices=('wt-primary','all-three'),default='wt-primary',
                     help='WT improvement is required by default; ED/NCR remain reported')
    args=parser.parse_args(argv)
    torch.set_num_threads(4)
    if hasattr(args,'limit_cases') and args.limit_cases is not None and args.limit_cases < 1:
        parser.error('--limit-cases must be positive')
    if args.command == 'merge-collections':
        return merge_collections(args.collections, args.output, dense=True)
    if args.command == 'merge-adapters':
        return adapter.merge_adapters(args.checkpoints, args.output)
    return globals()[args.command](args)


if __name__ == '__main__':
    main()
