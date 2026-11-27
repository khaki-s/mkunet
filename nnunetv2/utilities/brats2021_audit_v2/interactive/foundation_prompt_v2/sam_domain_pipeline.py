"""Supervised adaptation of pretrained SAM3, with GT-free self prompts.

Whole axial slices, official image resolution, original prompt/mask decoder.
GT is a loss/validation target only. Test inference and scoring are separate.
"""
from contextlib import nullcontext
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import time

import nibabel as nib
import numpy as np
from scipy.ndimage import distance_transform_edt
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

from .common import (manifest, read_json, write_json, file_identity, check_file,
                     read_labels, same_grid, save_mask, region_metrics)
from .sam_dense import SAM_SOURCE, SAM_CHECKPOINT, HEADS


def implementation():
    folder = Path(__file__).parent
    return {n: file_identity(folder/n)['sha256'] for n in
            ('sam_domain_pipeline.py', 'sam_domain_model.py', 'sam_domain_data.py')}


def segmentation_loss(logits, target):
    """Original SAM3 focal (20) + Dice (1), FP32; no GT-generated prompt."""
    from .sam_domain_model import sam_supervised_loss
    target = target.to(logits.device, dtype=torch.float32).reshape_as(logits)
    return sam_supervised_loss(logits, target)[0]


def baseline_priors(baseline, heads):
    result={}
    for head in heads:
        binary=baseline>HEADS.index(head)
        sdf=distance_transform_edt(binary)-distance_transform_edt(~binary)
        result[head]=(1/(1+np.exp(-np.clip(sdf/2,-20,20)))).astype(np.float32)
    return result


def compose(baseline, probabilities, alpha, *, priors=None):
    """Global blend strength, fixed on validation; never a per-patient oracle.

    For WT-only refinement the TC/ET masks are preserved exactly. This is a
    stated restricted experiment, not evidence of improved TC/ET segmentation.
    """
    original = np.asarray(baseline, dtype=np.uint8)
    if not 0 <= alpha <= 1:
        raise ValueError('alpha must be in [0,1]')
    if alpha == 0:
        return original.copy()
    if priors is None:
        priors=baseline_priors(original,probabilities)
    binary = [(original > h).copy() for h in range(3)]
    for head, p in probabilities.items():
        h = HEADS.index(head)
        if p.shape != original.shape or not np.isfinite(p).all() or p.min()<0 or p.max()>1:
            raise ValueError('Probability must be finite and on the original native grid')
        # Unprocessed empty slices have p=NaN nowhere: they carry the exact
        # original 0/1 probability and do not invent an unanchored lesion.
        prior = priors[head]
        binary[h] = ((1-alpha)*prior + alpha*p) >= .5
    # Preserve inner source regions when only WT was trained; for three heads
    # use standard union hierarchy, allowing inner evidence to enlarge WT.
    binary[1] |= binary[2]
    binary[0] |= binary[1]
    result = np.zeros_like(original)
    for h, mask in enumerate(binary):
        result[mask] = h+1
    return result


def _to_device(features, device):
    return {k: v.to(device=device, dtype=torch.float32, non_blocking=True)
            for k, v in features.items()}


def _amp(device):
    return torch.autocast('cuda', dtype=torch.float16) if str(device).startswith('cuda') else nullcontext()


def memory_preflight(device, required_gib):
    """Fail before loading rather than silently changing image dimensions."""
    if not str(device).startswith('cuda'):
        return
    free, total = torch.cuda.mem_get_info(device)
    if free < required_gib*2**30:
        raise RuntimeError(f'{device}: need {required_gib:g} GiB free before starting; '
                           f'only {free/2**30:.2f} GiB available. Input resolution is unchanged.')


def _case_rows(dataset):
    rows = {r['case_id']: r for r in dataset.source['cases']}
    groups = {}
    for i, record in enumerate(dataset.records):
        groups.setdefault(record['case_id'], []).append(i)
    # Include completed cache cases without any prompted slice (unchanged).
    for key in dataset.case_ids:
        groups.setdefault(key, [])
    return rows, groups


@torch.no_grad()
def infer_cases(model, dataset, device):
    """No GT accepted or opened in this inference generator."""
    if dataset.with_gt:
        raise ValueError('Inference requires a dataset with_gt=False')
    model.eval()
    rows, groups = _case_rows(dataset)
    for key, indices in groups.items():
        started = time.monotonic()
        row = rows[key]
        check_file(row['prediction'])
        reference = nib.load(row['prediction']['path'])
        baseline = read_labels(reference)
        probabilities = {h: (baseline > HEADS.index(h)).astype(np.float32)
                         for h in dataset.heads}
        for i in indices:
            item = dataset[i]
            with _amp(device):
                logits = model(_to_device(item['features'], device), item['baseline'])
            p = logits.float().sigmoid()[0, 0].cpu().numpy()
            if p.shape != baseline.shape[:2]:
                raise ValueError('SAM must return the complete native slice')
            probabilities[item['head']][:, :, item['z']] = p
        yield key, baseline, reference, probabilities, time.monotonic()-started


def summarize(records):
    output = {}
    for region in ('WT', 'TC', 'ET', 'ED', 'NCR'):
        rows = [r for r in records if r['region'] == region]
        valid = [r for r in rows if r['gain_pp'] is not None]
        gains = np.array([r['gain_pp'] for r in valid])
        output[region] = {'n': len(valid), 'total': len(rows),
            'baseline_pct': float(np.mean([r['baseline_pct'] for r in valid])) if valid else None,
            'final_pct': float(np.mean([r['final_pct'] for r in valid])) if valid else None,
            'mean_gain_pp': float(gains.mean()) if valid else None,
            'improved_fraction': float((gains > 1e-8).mean()) if valid else None,
            'worsened_fraction': float((gains < -1e-8).mean()) if valid else None,
            'empty_gt_added_fp': sum(r['empty_gt_added_fp'] for r in rows)}
        paired_defined=[r for r in valid if not r['both_empty_baseline'] and not r['both_empty_final']]
        output[region]['defined_pair_n']=len(paired_defined)
        output[region]['defined_pair_baseline_pct']=float(np.mean([r['baseline_pct'] for r in paired_defined])) if paired_defined else None
        output[region]['defined_pair_final_pct']=float(np.mean([r['final_pct'] for r in paired_defined])) if paired_defined else None
        output[region]['defined_pair_gain_pp']=float(np.mean([r['gain_pp'] for r in paired_defined])) if paired_defined else None
        if valid:
            rng=np.random.default_rng(20261117)
            means=gains[rng.integers(0,len(gains),size=(2000,len(gains)))].mean(1)
            output[region]['paired_bootstrap_95ci_gain_pp']=np.percentile(means,[2.5,97.5]).tolist()
        else:
            output[region]['paired_bootstrap_95ci_gain_pp']=None
    return output


def metric_rows(key, baseline, final, truth):
    before, after = region_metrics(baseline, truth), region_metrics(final, truth)
    result = []
    for region in ('WT', 'TC', 'ET', 'ED', 'NCR'):
        b, a = before[region]['dice'], after[region]['dice']
        # A both-empty region becoming FP is explicitly a failure, not dropped.
        b_for_delta = 1. if b is None else b
        a_for_delta = 1. if a is None else a
        result.append({'case_id': key, 'region': region,
            'baseline_pct': b_for_delta*100, 'final_pct': a_for_delta*100,
            'gain_pp': (a_for_delta-b_for_delta)*100,
            'both_empty_baseline': b is None, 'both_empty_final': a is None,
            'empty_gt_added_fp': int(b is None and a is not None)})
    return result


def validate(model, dataset, gt_manifest, device, alphas):
    """Validation targets rank GLOBAL protocols only; no backprop or patient choice."""
    truth_source = manifest(gt_manifest, allowed=('validation',))
    truth_rows = {r['case_id']: r for r in truth_source['cases']}
    if dataset.source['split'] != 'validation':
        raise ValueError('Checkpoint selection uses validation only')
    if dataset.source['splits_file'] != truth_source['splits_file']:
        raise ValueError('Validation split identity differs')
    records = {str(a): [] for a in alphas}
    seconds = 0.
    for key, baseline, reference, probabilities, elapsed in infer_cases(model, dataset, device):
        row = truth_rows[key]
        clean = next(r for r in dataset.source['cases'] if r['case_id'] == key)
        if {k:v for k,v in row.items() if k!='gt'} != {k:v for k,v in clean.items() if k!='gt'}:
            raise ValueError('Validation image/mask input mismatch')
        check_file(row['gt'])
        gt_image = nib.load(row['gt']['path']); same_grid(reference, gt_image)
        truth = read_labels(gt_image)
        priors=baseline_priors(baseline,probabilities)
        for a in alphas:
            records[str(a)].extend(metric_rows(key, baseline, compose(baseline, probabilities, a,priors=priors), truth))
        seconds += elapsed
    candidates = [{'alpha': a, 'regions': summarize(records[str(a)])} for a in alphas]
    # First require no mean TC/ET loss. Among those choose greatest mean WT gain.
    safe = [c for c in candidates if all(c['regions'][h]['mean_gain_pp'] is not None and
              c['regions'][h]['mean_gain_pp'] >= -1e-8 for h in ('TC', 'ET'))]
    best = max(safe or candidates, key=lambda c: (c['regions']['WT']['mean_gain_pp'], -c['alpha']))
    return {'selected': best, 'candidates': candidates, 'case_count': len(dataset.case_ids),
            'full_val80': set(dataset.case_ids) == set(truth_rows) and len(truth_rows) == 80,
            'seconds': seconds, 'gt_only_for_scoring': True}


def cache(args):
    from .sam_domain_model import FrozenSamDomainEncoder
    from .sam_domain_data import cache_cases
    source = manifest(args.manifest)
    if source['split'] != 'train':
        manifest(args.manifest, deployment=True)
    torch.set_num_threads(4)
    memory_preflight(args.device, 8.)
    encoder = FrozenSamDomainEncoder(args.source, args.sam_checkpoint, args.device)
    from .sam_domain_model import source_identity
    contract = {'sam_checkpoint': file_identity(args.sam_checkpoint),
        'sam_source': str(Path(args.source).resolve()), 'encoder_resolution': 1008,
        'spatial_crop': False, 'sam_source_sha256': source_identity(args.source),
        'implementation': {k:v for k,v in implementation().items() if k!='sam_domain_pipeline.py'}}
    return cache_cases(args.manifest, args.output, encoder.encode, contract=contract,
                       heads=tuple(args.heads.split(',')), limit_cases=args.limit_cases,
                       num_shards=args.num_shards, shard_index=args.shard_index)


def train(args):
    from .sam_domain_data import CachedSliceDataset
    from .sam_domain_model import build_domain_decoder
    if args.epochs < 1 or args.accumulate < 1 or args.lr <= 0:
        raise ValueError('epochs, accumulation and lr must be positive')
    world = int(os.environ.get('WORLD_SIZE', '1')); rank = int(os.environ.get('RANK', '0'))
    local = int(os.environ.get('LOCAL_RANK', '0'))
    device = f'cuda:{local}' if torch.cuda.is_available() else 'cpu'
    memory_preflight(device, 4.)
    if world > 1:
        if device != 'cpu': torch.cuda.set_device(local)
        dist.init_process_group('nccl' if device != 'cpu' else 'gloo',
                                **({'device_id':torch.device(device)} if device!='cpu' else {}))
    torch.set_num_threads(args.cpu_threads)
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    training = CachedSliceDataset(args.train_cache, with_gt=True, allowed_splits=('train',))
    validation = CachedSliceDataset(args.val_cache, with_gt=False, allowed_splits=('validation',))
    if set(training.case_ids) & set(validation.case_ids): raise ValueError('Patient overlap')
    if training.source['initial_checkpoint'] != validation.source['initial_checkpoint']:
        raise ValueError('Training/validation initial nnUNet checkpoint differs')
    if tuple(training.heads) != tuple(validation.heads): raise ValueError('Training/validation heads differ')
    if training.index['binding']['contract'] != validation.index['binding']['contract']: raise ValueError('Encoder contracts differ')
    from .sam_domain_model import source_identity
    encoder_contract=training.index['binding']['contract']
    expected={'sam_checkpoint':file_identity(args.sam_checkpoint),
        'sam_source':str(Path(args.source).resolve()),'encoder_resolution':1008,
        'spatial_crop':False,'sam_source_sha256':source_identity(args.source),
        'implementation':{k:v for k,v in implementation().items() if k!='sam_domain_pipeline.py'}}
    if encoder_contract!=expected:
        raise ValueError('Cached encoder weights/source differ from requested SAM training initialization')
    out = Path(args.output)
    if rank == 0:
        if out.exists(): raise FileExistsError('Use a fresh training directory')
        out.mkdir(parents=True)
    if world > 1: dist.barrier()
    model = build_domain_decoder(args.source, checkpoint=args.sam_checkpoint, mode=args.mode, lora_rank=args.lora_rank).to(device)
    configuration = model.configuration()
    bare = model
    if world > 1:
        model = DistributedDataParallel(model, device_ids=[local] if device != 'cpu' else None,
                                        find_unused_parameters=True)
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=.01)
    scaler = torch.amp.GradScaler('cuda', enabled=device != 'cpu', init_scale=1024.)
    sampler = DistributedSampler(training, num_replicas=world, rank=rank, shuffle=True, seed=args.seed)
    loader = DataLoader(training, batch_size=None, sampler=sampler, num_workers=args.workers)
    provenance = {'schema': 'sam_domain_checkpoint_v1', 'train_cache': file_identity(args.train_cache),
        'val_cache': file_identity(args.val_cache), 'val_manifest': file_identity(args.val_manifest),
        'implementation': implementation(), 'configuration': configuration,
        'sam_checkpoint': file_identity(args.sam_checkpoint), 'source': args.source,
        'train_case_ids': training.case_ids, 'heads': training.heads,
        'initial_checkpoint':training.source['initial_checkpoint'],
        'full_train320': len(training.case_ids) == 320,
        'world_size': world, 'accumulate_per_rank': args.accumulate,
        'effective_batch': world*args.accumulate, 'input_resolution': 1008,
        'native_slice_cropped': False, 'seed': args.seed,
        'loss': 'official SAM3 focal(20) + Dice(1)',
        'train_initial_masks': 'in-sample; OOF masks recommended for a later ablation',
        'encoder_contract': training.index['binding']['contract']}
    history = []; best = -math.inf
    for epoch in range(args.epochs):
        sampler.set_epoch(epoch); model.train(); optimizer.zero_grad(set_to_none=True)
        total = 0.; steps = 0; updates = 0; skipped = 0
        for step, item in enumerate(loader):
            group_start = (step//args.accumulate)*args.accumulate
            group_size = min(args.accumulate, len(loader)-group_start)
            sync = (step+1) % args.accumulate == 0 or step+1 == len(loader)
            context = model.no_sync() if world > 1 and not sync else nullcontext()
            with context:
                with _amp(device):
                    # DDP moves Tensor arguments onto its CUDA device. Prompt
                    # geometry deliberately uses CPU NumPy, so hand it an
                    # ndarray rather than a CPU Tensor that DDP would migrate.
                    native_mask=np.asarray(item['baseline'])
                    logits = model(_to_device(item['features'], device), native_mask)
                loss = segmentation_loss(logits, item['target'])
                if not torch.isfinite(loss): raise FloatingPointError('Nonfinite decoder loss')
                scaler.scale(loss/group_size).backward()
            total += float(loss.detach()); steps += 1
            if sync:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(parameters, 1.)
                scale = scaler.get_scale(); scaler.step(optimizer); scaler.update()
                overflow = scaler.get_scale() < scale
                skipped += int(overflow); updates += int(not overflow)
                optimizer.zero_grad(set_to_none=True)
            if rank==0 and (step==0 or (step+1)%100==0):
                print(json.dumps({'stage':'train','epoch':epoch+1,'rank_step':step+1,
                    'rank_steps':len(loader),'running_loss':total/steps,
                    'optimizer_steps':updates,'amp_skips':skipped}),flush=True)
        totals = torch.tensor([total, steps, updates, skipped], dtype=torch.float64, device=device)
        if world > 1: dist.all_reduce(totals)
        if rank == 0:
            measured = validate(bare, validation, args.val_manifest, device, (0., .25, .5, .75, 1.))
            gain = measured['selected']['regions']['WT']['mean_gain_pp']
            entry = {'epoch': epoch+1, 'train_loss': float(totals[0]/totals[1]),
                'optimizer_steps_across_ranks': int(totals[2]), 'overflow_skips_across_ranks': int(totals[3]),
                'validation': measured, 'peak_cuda_allocated_gib': torch.cuda.max_memory_allocated()/2**30 if device != 'cpu' else 0.}
            history.append(entry)
            state = {'metadata': {**provenance, 'epoch': epoch+1, 'validation': measured},
                     'state_dict': {k: v.detach().cpu() for k,v in bare.state_dict().items()},
                     'optimizer': optimizer.state_dict(), 'scaler': scaler.state_dict()}
            torch.save(state, out/'checkpoint_last.pth')
            if gain > best:
                best = gain; torch.save(state, out/'checkpoint_best.pth')
            write_json(out/'history.json', history)
            print(json.dumps({'epoch': epoch+1, 'train_loss': entry['train_loss'],
                'val_WT_gain_pp': gain, 'alpha': measured['selected']['alpha'],
                'val_cases': measured['case_count'], 'peak_gib': entry['peak_cuda_allocated_gib']}), flush=True)
        if world > 1: dist.barrier()
    digest=hashlib.sha256()
    for key,value in sorted(bare.state_dict().items()):
        digest.update(key.encode());digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    checksum=digest.hexdigest()
    hashes=[checksum]
    if world>1:
        hashes=[None]*world;dist.all_gather_object(hashes,checksum)
        if len(set(hashes))!=1:raise AssertionError('DDP final model replicas differ')
    write_json(out/f'resources_rank{rank}.json',{'rank':rank,'world_size':world,
        'device':device,'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30 if device!='cpu' else 0.,
        'peak_reserved_gib':torch.cuda.max_memory_reserved()/2**30 if device!='cpu' else 0.,
        'model_sha256':checksum,'replicas_identical':len(set(hashes))==1,
        'official_encoder_resolution':1008,'native_slice_cropped':False})
    if world > 1: dist.destroy_process_group()
    return out/'checkpoint_best.pth'


def predict(args):
    from .sam_domain_data import CachedSliceDataset
    from .sam_domain_model import build_domain_decoder
    data = CachedSliceDataset(args.cache, with_gt=False, allowed_splits=('validation', 'test'))
    manifest(data.index['binding']['source']['path'], deployment=True)
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=True); meta = state['metadata']
    if meta['implementation'] != implementation(): raise ValueError('Checkpoint/source mismatch')
    if meta['encoder_contract'] != data.index['binding']['contract']: raise ValueError('Image encoder/cache mismatch')
    if meta['initial_checkpoint'] != data.source['initial_checkpoint']:
        raise ValueError('Deployment initial nnUNet checkpoint differs from training')
    if tuple(meta['heads']) != tuple(data.heads): raise ValueError('Head mismatch')
    selected = meta['validation']['selected']; alpha = selected['alpha']
    if data.source['split'] == 'test':
        if not (meta['full_train320'] and meta['validation']['full_val80'] and
                selected['regions']['WT']['mean_gain_pp'] > 0):
            raise ValueError('Test requires trained320 and positive WT on complete val80')
        if any(selected['regions'][h]['mean_gain_pp'] < -1e-8 for h in ('TC', 'ET')):
            raise ValueError('Selected validation protocol regressed TC/ET')
    memory_preflight(args.device, 4.)
    model = build_domain_decoder(meta['source'], configuration=meta['configuration']).to(args.device)
    model.load_state_dict(state['state_dict'], strict=True); model.requires_grad_(False)
    out = Path(args.output)
    if out.exists(): raise FileExistsError('Use a fresh prediction directory')
    out.mkdir(parents=True); completed = {}
    binding = {'schema': 'sam_domain_prediction_v1', 'checkpoint': file_identity(args.checkpoint),
        'cache': file_identity(args.cache), 'split': data.source['split'], 'alpha': alpha,
        'implementation': implementation(), 'gt_used': False, 'heads': data.heads}
    for key, baseline, reference, probabilities, seconds in infer_cases(model, data, args.device):
        final = compose(baseline, probabilities, alpha)
        completed[key] = {'baseline': save_mask(out/key/'baseline.nii.gz', baseline, reference),
            'final': save_mask(out/key/'final.nii.gz', final, reference), 'seconds': seconds}
        write_json(out/'predictions.json', {'binding': binding, 'completed': completed})
    return out/'predictions.json'


def score(args):
    prediction = read_json(args.predictions); source = manifest(args.manifest, allowed=('validation','test'))
    if prediction['binding']['split'] != source['split']: raise ValueError('Score split mismatch')
    check_file(prediction['binding']['cache'])
    cached=read_json(prediction['binding']['cache']['path'])
    check_file(cached['binding']['source'])
    deployed=manifest(cached['binding']['source']['path'],deployment=True)
    source_inputs={r['case_id']:{k:v for k,v in r.items() if k!='gt'} for r in source['cases']}
    if source_inputs!={r['case_id']:r for r in deployed['cases']}:
        raise ValueError('Scoring source images/masks differ from deployed inputs')
    lookup = {r['case_id']:r for r in source['cases']}; records=[]
    for key, row in prediction['completed'].items():
        original = lookup[key]
        for identity in (row['baseline'], row['final'], original['gt'], original['prediction']): check_file(identity)
        base_image=nib.load(row['baseline']['path']); final_image=nib.load(row['final']['path']); gt_image=nib.load(original['gt']['path'])
        same_grid(base_image, final_image); same_grid(base_image, gt_image)
        baseline=read_labels(base_image)
        if not np.array_equal(baseline, read_labels(nib.load(original['prediction']['path']))): raise ValueError('Baseline differs')
        records.extend(metric_rows(key, baseline, read_labels(final_image), read_labels(gt_image)))
    out=Path(args.output); out.mkdir(parents=True,exist_ok=True)
    summary={'schema':'sam_domain_score_v1','split':source['split'],'cases':len(prediction['completed']),
        'full_cohort':set(prediction['completed'])==set(lookup),'regions':summarize(records),
        'predictions':file_identity(args.predictions), 'manifest':file_identity(args.manifest),
        'metric_convention':'both-empty=100%, includes absent-class false-positive penalties'}
    write_json(out/'score.json',summary)
    with (out/'per_case.csv').open('w') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(records[0]));writer.writeheader()
        writer.writerows({k:f'{v:.3f}' if isinstance(v,float) else v for k,v in r.items()} for r in records)
    return out/'score.json'


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__); sub=p.add_subparsers(dest='command',required=True)
    q=sub.add_parser('cache');q.add_argument('--manifest',required=True);q.add_argument('--output',required=True)
    q.add_argument('--source',default=SAM_SOURCE);q.add_argument('--sam-checkpoint',default=SAM_CHECKPOINT)
    q.add_argument('--device',default='cuda:0');q.add_argument('--heads',default='WT');q.add_argument('--limit-cases',type=int,default=0)
    q.add_argument('--num-shards',type=int,default=1);q.add_argument('--shard-index',type=int,default=0)
    q=sub.add_parser('merge-cache');q.add_argument('--caches',nargs='+',required=True);q.add_argument('--output',required=True)
    q=sub.add_parser('train');q.add_argument('--train-cache',required=True);q.add_argument('--val-cache',required=True)
    q.add_argument('--val-manifest',required=True);q.add_argument('--output',required=True)
    q.add_argument('--source',default=SAM_SOURCE);q.add_argument('--sam-checkpoint',default=SAM_CHECKPOINT)
    q.add_argument('--mode',choices=('decoder','qv_lora'),default='decoder');q.add_argument('--lora-rank',type=int,default=4)
    q.add_argument('--epochs',type=int,default=30);q.add_argument('--lr',type=float,default=1e-5)
    q.add_argument('--accumulate',type=int,default=8);q.add_argument('--workers',type=int,default=0)
    q.add_argument('--cpu-threads',type=int,default=4);q.add_argument('--seed',type=int,default=20261117)
    q=sub.add_parser('predict');q.add_argument('--cache',required=True);q.add_argument('--checkpoint',required=True)
    q.add_argument('--output',required=True);q.add_argument('--device',default='cuda:0')
    q=sub.add_parser('score');q.add_argument('--predictions',required=True);q.add_argument('--manifest',required=True);q.add_argument('--output',required=True)
    args=p.parse_args(argv)
    if args.command=='merge-cache':
        from .sam_domain_data import merge_caches
        return merge_caches(args.caches,args.output)
    return globals()[args.command](args)
