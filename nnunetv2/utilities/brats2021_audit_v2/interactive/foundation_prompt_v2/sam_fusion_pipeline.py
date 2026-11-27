"""SAM3 self-prompted dense candidates + GT-supervised native-volume Atlas gates."""
import argparse
import csv
import gc
from pathlib import Path
import time

import nibabel as nib
import numpy as np
import torch

from . import sam_dense, sam_full_fusion
from . import atlas_pipeline as atlas
from .common import (manifest,load_case,read_json,write_json,file_identity,check_file,
                     save_mask,read_labels,same_grid)


def contract(source):
    check_file(source['initial_checkpoint'])
    folder=Path(__file__).parent
    sam_folder=Path(sam_dense.SAM_SOURCE)/'sam3'
    return {'method':'sam3_selfprompt_fullvolume_atlas_v1','sam_checkpoint':file_identity(sam_dense.SAM_CHECKPOINT),
            'sam_source':{str(p.relative_to(sam_folder)):file_identity(p)['sha256'] for p in sorted(sam_folder.rglob('*.py'))},
            'candidate_code':{n:file_identity(folder/n)['sha256'] for n in ('sam_dense.py','sam3_candidates.py')},
            'initial_checkpoint':source['initial_checkpoint'],'splits':source['splits_file'],
            'encoder_resolution':1008,'medical_volume':'native; no resize or spatial crop',
            'modality_by_head':sam_dense.MODALITIES,'variants':list(sam_dense.VARIANTS),
            'gt_in_prompts':False,'gt_usage':'train gate labels and separate scoring only'}


def _pair_indexes(root,index,binding,completed):
    for variant in sam_dense.VARIANTS:
        write_json(root/f'pairs_{variant}.json',{'schema':'sam_fullvolume_pairs_v1','split':'train',
                   'source_manifest':binding['source'],'collection':file_identity(index),
                   'sam_contract':binding['contract'],'variant':variant,
                   'cases':[r for c in completed.values() for r in c['pairs'][variant]],
                   'full_train320':len(completed)==320})


def collect(args):
    source=manifest(args.manifest,allowed=('train',))
    method=contract(source)
    rows=source['cases'][:args.limit_cases] if args.limit_cases else source['cases']
    if args.num_shards<1 or not 0<=args.shard_index<args.num_shards:
        raise ValueError('Invalid case partition')
    rows=rows[args.shard_index::args.num_shards]
    root=Path(args.output);root.mkdir(parents=True,exist_ok=True)
    index=root/'collection.json'
    binding={'schema':'sam_dense_collect_v1','source':file_identity(args.manifest),'contract':method,
             'limit_cases':args.limit_cases,'num_shards':args.num_shards,'shard_index':args.shard_index}
    completed=atlas._resume(index,binding)
    engine=None
    for row in rows:
        key=row['case_id']
        if key in completed:continue
        started=time.monotonic()
        mri,current,reference=load_case({k:v for k,v in row.items() if k!='gt'})
        if row.get('availability',[1]*4)!=[1]*4:raise ValueError('This run requires four available modalities')
        if engine is None:engine=sam_dense.SamDenseRefiner(args.device)
        generated,details=engine.generate(mri,current)  # No GT argument or access.
        folder=root/key;folder.mkdir(exist_ok=True)
        identities=[];pairs={variant:[] for variant in sam_dense.VARIANTS}
        for variant,heads in generated.items():
            for h,head in enumerate(sam_dense.HEADS):
                path=folder/f'{variant}_{head}.nii.gz'
                identities.append(save_mask(path,heads[head],reference))
        # Only after finishing every frozen-SAM call may training labels load.
        check_file(row['gt']);truth=nib.load(row['gt']['path']);same_grid(truth,reference)
        image_path=folder/'training_image.npz'
        np.savez_compressed(image_path,mri=mri,baseline=current,gt=read_labels(truth))
        image_id=file_identity(image_path);identities.append(image_id)
        details_id=folder/'sam_calls.json';write_json(details_id,details);identities.append(file_identity(details_id))
        for variant in sam_dense.VARIANTS:
            for h,head in enumerate(sam_dense.HEADS):
                pairs[variant].append({'case_id':key,'head':h,'image':image_id,
                                       'candidate':file_identity(folder/f'{variant}_{head}.nii.gz')})
        completed[key]={'files':identities,'pairs':pairs,'sam_calls':details['sam_predict_calls'],
                        'native_shape':list(current.shape),'seconds':time.monotonic()-started}
        write_json(index,{'binding':binding,'completed':completed,'full_cohort':len(completed)==len(source['cases'])})
        _pair_indexes(root,index,binding,completed)
        print(f'SAM collect {key} {len(completed)}/{len(rows)}',flush=True)
        del mri,current,generated;gc.collect()
        if torch.cuda.is_available():torch.cuda.empty_cache()
    if not index.exists():write_json(index,{'binding':binding,'completed':completed,'full_cohort':False})
    _pair_indexes(root,index,binding,completed)
    return index


def merge(args):
    root=Path(args.output)
    if root.exists():raise FileExistsError('Use a new merged collection directory')
    completed={};binding=None;parts=set()
    for path in args.collections:
        data=read_json(path);current=dict(data['binding']);part=current.pop('shard_index')
        if binding is None:binding=current
        if current!=binding or part in parts:raise ValueError('Mismatched or duplicate SAM collection shard')
        check_file(current['source']);source=manifest(current['source']['path'],allowed=('train',))
        rows=source['cases'][:current['limit_cases']] if current['limit_cases'] else source['cases']
        expected={r['case_id'] for r in rows[part::current['num_shards']]}
        if set(data['completed'])!=expected:raise ValueError('SAM shard is incomplete or has foreign cases')
        for case in data['completed'].values():
            for identity in case['files']:check_file(identity)
        if completed.keys()&data['completed'].keys():raise ValueError('Overlapping SAM cases')
        completed.update(data['completed']);parts.add(part)
    if binding is None or parts!=set(range(binding['num_shards'])):raise ValueError('All SAM collection shards required')
    binding['merged_from']=[file_identity(p) for p in args.collections]
    root.mkdir(parents=True);index=root/'collection.json'
    write_json(index,{'binding':binding,'completed':completed,'full_cohort':len(completed)==len(source['cases'])})
    _pair_indexes(root,index,binding,completed)
    return index


def predict(args):
    source=manifest(args.manifest,deployment=True,allowed=('validation','test'))
    method=contract(source)
    checkpoints=[Path(args.checkpoints)/f'head{h}/checkpoint.pth' for h in range(3)]
    states=[torch.load(q,map_location='cpu',weights_only=True)['metadata'] for q in checkpoints]
    if any(s['sam_contract']!=method or s['variant']!=states[0]['variant'] for s in states):
        raise ValueError('SAM prediction protocol differs from trained gates')
    if source['split']=='test':
        locked=read_json(args.validation_lock) if args.validation_lock else {}
        if (not locked.get('passed') or locked.get('sam_contract')!=method
                or locked.get('checkpoints')!=[file_identity(p) for p in checkpoints]):
            raise ValueError('Test requires a passing independent val80 lock of this exact SAM method')
        check_file(locked['score'])
    root=Path(args.output);root.mkdir(parents=True,exist_ok=True)
    binding={'schema':'sam_fullvolume_predictions_v1','source':file_identity(args.manifest),
             'contract':method,'checkpoints':[file_identity(q) for q in checkpoints],
             'split':source['split'],'limit_cases':args.limit_cases}
    index=root/'predictions.json';completed=atlas._resume(index,binding)
    rows=source['cases'][:args.limit_cases] if args.limit_cases else source['cases']
    for row in rows:
        key=row['case_id']
        if key in completed:continue
        started=time.monotonic();mri,current,reference=load_case(row)
        engine=sam_dense.SamDenseRefiner(args.device)
        generated,details=engine.generate(mri,current)
        del engine;gc.collect();torch.cuda.empty_cache()
        probabilities=[sam_full_fusion.predict(checkpoints[h],mri,current,generated[states[0]['variant']][head],h,
                       device=args.device) for h,head in enumerate(sam_dense.HEADS)]
        final=sam_full_fusion.project(probabilities)
        folder=root/key
        before=save_mask(folder/'baseline.nii.gz',current,reference)
        after=save_mask(folder/'final.nii.gz',final,reference)
        completed[key]={'files':[before,after],'baseline':before,'final':after,'sam_calls':details['sam_predict_calls'],
                        'seconds':time.monotonic()-started,'gt_used':False}
        write_json(index,{'binding':binding,'completed':completed,'full_cohort':len(completed)==len(source['cases'])})
        print(f'SAM predict {key}',flush=True)
    return index


def score(args):
    data=read_json(args.predictions);binding=data['binding']
    if binding['schema']!='sam_fullvolume_predictions_v1':raise ValueError('Expected SAM full-volume predictions')
    check_file(binding['source'])
    for identity in binding['checkpoints']:check_file(identity)
    source=manifest(binding['source']['path'],deployment=True)
    truth=manifest(args.manifest,allowed=(source['split'],));rows={r['case_id']:r for r in truth['cases']}
    clean={r['case_id']:r for r in source['cases']}
    records=[]
    for key,case in data['completed'].items():
        if {k:v for k,v in rows[key].items() if k!='gt'}!=clean[key]:raise ValueError('Scoring inputs differ')
        for identity in case['files']:check_file(identity)
        check_file(rows[key]['gt']);reference=nib.load(rows[key]['gt']['path']);gt=read_labels(reference)
        old,new=[nib.load(case[k]['path']) for k in ('baseline','final')]
        same_grid(old,reference);same_grid(new,reference)
        b,a=atlas.region_metrics(read_labels(old),gt),atlas.region_metrics(read_labels(new),gt)
        for head in atlas.REGIONS:
            before,after=b[head]['dice'],a[head]['dice']
            records.append({'case_id':key,'region':head,'baseline_pct':None if before is None else before*100,
                            'final_pct':None if after is None else after*100,
                            'gain_pp':None if before is None or after is None else 100*(after-before)})
    means={}
    for h in atlas.REGIONS:
        chosen=[r for r in records if r['region']==h and r['gain_pp'] is not None]
        means[h]={'n':len(chosen),'mean_gain_pp':float(np.mean([r['gain_pp'] for r in chosen])) if chosen else None}
    result={'schema':'sam_fullvolume_score_v1','split':source['split'],'full_cohort':data['full_cohort'],
            'case_count':len(data['completed']),'regions':means,'per_case':records,
            'predictions':file_identity(args.predictions),'sam_contract':binding['contract'],'checkpoints':binding['checkpoints']}
    path=Path(args.output)/'score.json';write_json(path,result);return path


def lock(args):
    score=read_json(args.score)
    if score.get('schema')!='sam_fullvolume_score_v1' or score['split']!='validation' or not score['full_cohort'] or score['case_count']!=80:
        raise ValueError('Need a full independent val80 evaluation')
    check_file(score['predictions'])
    for q in score['checkpoints']:check_file(q)
    gain=score['regions']['WT']['mean_gain_pp']
    if gain is None or gain<=1e-6:raise ValueError('WT did not improve on independent val80')
    if Path(args.output).exists():raise FileExistsError(args.output)
    write_json(args.output,{'passed':True,'score':file_identity(args.score),'sam_contract':score['sam_contract'],
                           'checkpoints':score['checkpoints'],'criterion':'mean WT Dice gain > 0'})
    return args.output


def diagnose(args):
    """Seen-training-case measurement, including an explicit no-SAM ablation."""
    pairs=read_json(args.pairs)
    if pairs.get('schema')!='sam_fullvolume_pairs_v1' or pairs['split']!='train':
        raise ValueError('Training diagnostics require train-only SAM pairs')
    check_file(pairs['source_manifest']);manifest(pairs['source_manifest']['path'],allowed=('train',))
    checkpoints=[Path(args.checkpoints)/f'head{h}/checkpoint.pth' for h in range(3)]
    states=[torch.load(q,map_location='cpu',weights_only=True)['metadata'] for q in checkpoints]
    if any(s['pairs']!=file_identity(args.pairs) for s in states):raise ValueError('Diagnostic data differ from training')
    grouped={}
    for row in pairs['cases']:grouped.setdefault(row['case_id'],{})[row['head']]=row
    output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    records=[];files=[]
    for key,heads in grouped.items():
        if set(heads)!={0,1,2}:raise ValueError('Need three heads for joint diagnostics')
        image=heads[0]['image'];check_file(image)
        with np.load(image['path'],allow_pickle=False) as z:mri,current=z['mri'],z['baseline']
        sam=[]
        for h in range(3):
            check_file(heads[h]['candidate'])
            sam.append(np.asanyarray(nib.load(heads[h]['candidate']['path']).dataobj)>0)
        probabilities=[sam_full_fusion.predict(checkpoints[h],mri,current,sam[h],h,device=args.device) for h in range(3)]
        final=sam_full_fusion.project(probabilities)
        ablated=sam_full_fusion.project([sam_full_fusion.predict(checkpoints[h],mri,current,current>h,h,
                                         device=args.device) for h in range(3)])
        if not np.array_equal(ablated,current):raise AssertionError('No-SAM ablation should preserve the source mask')
        reference=nib.load(heads[0]['candidate']['path'])
        files.extend([save_mask(output/key/'final.nii.gz',final,reference),
                      save_mask(output/key/'without_sam.nii.gz',ablated,reference)])
        # Predictions and ablation are fixed and saved before labels load.
        with np.load(image['path'],allow_pickle=False) as z:gt=z['gt']
        old,new=atlas.region_metrics(current,gt),atlas.region_metrics(final,gt)
        for region in atlas.REGIONS:
            b=100*(1 if old[region]['dice'] is None else old[region]['dice'])
            a=100*(1 if new[region]['dice'] is None else new[region]['dice'])
            row={'case_id':key,'region':region,'baseline_pct':b,'final_pct':a,'gain_pp':a-b,'without_sam_gain_pp':0.}
            if region in sam_dense.HEADS:
                h=sam_dense.HEADS.index(region);base=current>h;truth=gt>h
                dice=lambda p:100*(2*np.count_nonzero(p&truth)+1e-6)/(np.count_nonzero(p)+np.count_nonzero(truth)+1e-6)
                row.update(sam_candidate_pct=float(dice(sam[h])),
                           oracle_pct=float(dice(np.where(base==truth,base,sam[h]))),
                           changed_voxels=int(np.count_nonzero((final>h)!=base)))
            records.append(row)
        print(f'SAM training diagnostic {key}',flush=True)
    means={h:{k:float(np.mean([r[k] for r in records if r['region']==h])) for k in
              ('baseline_pct','final_pct','gain_pp','without_sam_gain_pp')} for h in atlas.REGIONS}
    result={'train_only':True,'case_count':len(grouped),'validation_cases':0,'test_cases':0,
            'not_generalization_evidence':True,'sam_variant':pairs['variant'],'sam_encoder_resolution':1008,
            'fusion_input':'entire native volume; no crop or resize','sam_required_for_every_nonzero_gain':True,
            'checkpoints':[file_identity(q) for q in checkpoints],'selected_epochs':[s['selected_epoch'] for s in states],
            'native_training_shapes':[s['native_training_shapes'] for s in states],
            'regions':means,'per_case':records,'prediction_files':files}
    write_json(output/'report.json',result)
    columns=list(dict.fromkeys(k for r in records for k in r))
    with (output/'per_case.csv').open('w') as handle:
        writer=csv.DictWriter(handle,fieldnames=columns);writer.writeheader();writer.writerows(records)
    print(means,flush=True)
    return output/'report.json'


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    q=sub.add_parser('collect');q.add_argument('--manifest',required=True);q.add_argument('--output',required=True)
    q.add_argument('--limit-cases',type=int);q.add_argument('--num-shards',type=int,default=1);q.add_argument('--shard-index',type=int,default=0)
    q.add_argument('--device',default='cuda:0')
    q=sub.add_parser('merge');q.add_argument('--collections',nargs='+',required=True);q.add_argument('--output',required=True)
    q=sub.add_parser('train');q.add_argument('--pairs',required=True);q.add_argument('--output',required=True)
    q.add_argument('--head',type=int,choices=(0,1,2),required=True);q.add_argument('--epochs',type=int,default=30)
    q.add_argument('--cache-cases',type=int,default=4);q.add_argument('--device',default='cuda:0')
    q=sub.add_parser('predict');q.add_argument('--manifest',required=True);q.add_argument('--checkpoints',required=True)
    q.add_argument('--output',required=True);q.add_argument('--device',default='cuda:0');q.add_argument('--limit-cases',type=int)
    q.add_argument('--validation-lock')
    q=sub.add_parser('score');q.add_argument('--predictions',required=True);q.add_argument('--manifest',required=True);q.add_argument('--output',required=True)
    q=sub.add_parser('lock');q.add_argument('--score',required=True);q.add_argument('--output',required=True)
    q=sub.add_parser('diagnose');q.add_argument('--pairs',required=True);q.add_argument('--checkpoints',required=True)
    q.add_argument('--output',required=True);q.add_argument('--device',default='cuda:0')
    args=p.parse_args(argv);torch.set_num_threads(4)
    if args.command=='train':return sam_full_fusion.train(args.pairs,args.output,args.head,epochs=args.epochs,device=args.device,cache_cases=args.cache_cases)
    return globals()[args.command](args)
