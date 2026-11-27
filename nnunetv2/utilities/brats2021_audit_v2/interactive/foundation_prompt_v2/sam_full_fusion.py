"""Original Atlas gate trained/inferred on entire native volumes, with SAM candidates."""
from collections import OrderedDict
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from torch.utils.checkpoint import checkpoint as recompute

from . import atlas_adapter as atlas
from .common import read_json, write_json, file_identity, check_file, manifest


def load_training_pair(row):
    check_file(row['image']);check_file(row['candidate'])
    with np.load(row['image']['path'],allow_pickle=False) as z:
        mri,base,gt=z['mri'],z['baseline'],z['gt']
    proposed=np.asanyarray(nib.load(row['candidate']['path']).dataobj)>0
    head=int(row['head'])
    return mri,base>head,proposed,gt>head


def features(row,cache_dir=None):
    mri,base,sam,gt=load_training_pair(row)
    cache_file=marker=None
    if cache_dir is not None:
        key=hashlib.sha256(json.dumps({'image':row['image']['sha256'],'candidate':row['candidate']['sha256'],
            'head':row['head'],'atlas':atlas.SOURCE_SHA256},sort_keys=True).encode()).hexdigest()
        cache_file=Path(cache_dir)/f'{key}.npz';marker=cache_file.with_suffix('.json')
    hit=False
    if marker is not None and marker.exists() and cache_file.exists():
        try:
            check_file(read_json(marker)['file'])
            with np.load(cache_file,allow_pickle=False) as z:
                sam_soft=torch.from_numpy(z['sam_soft']);base_soft=torch.from_numpy(z['base_soft'])
            hit=True
        except (ValueError,OSError):
            hit=False  # Rebuild only derived features from verified raw inputs.
    if hit:
        image=torch.from_numpy(np.ascontiguousarray(mri,dtype=np.float32))[None]
        if sam_soft.shape!=(1,1,*base.shape) or base_soft.shape!=sam_soft.shape:
            raise ValueError('Full native SDF cache shape mismatch')
    else:
        image,sam_soft,base_soft=atlas.make_features(mri,base,sam)
        if cache_file is not None:
            cache_file.parent.mkdir(parents=True,exist_ok=True)
            temporary=cache_file.with_suffix('.tmp.npz')
            np.savez_compressed(temporary,sam_soft=sam_soft.numpy(),base_soft=base_soft.numpy())
            temporary.replace(cache_file)
            write_json(marker,{'file':file_identity(cache_file),'dtype':'float32','lossless':True,
                               'native_shape':list(base.shape)})
    binary=lambda a:torch.from_numpy(np.ascontiguousarray(a,dtype=np.float32))[None,None]
    target,valid=atlas._upstream()['build_oracle_gate'](binary(sam),binary(base),binary(gt))
    return image,sam_soft,base_soft,binary(gt),target,valid


def _model(head,device):
    torch.manual_seed(20261118+head)
    return atlas.get_adapter(device=device)


def project(probabilities):
    """WT is the primary head; constrain inner classes without enlarging WT."""
    p=np.asarray(probabilities,dtype=np.float32)
    if p.ndim!=4 or not np.isfinite(p).all() or p.min()<0 or p.max()>1:
        raise ValueError('Expected three finite native probability volumes')
    p=p.copy();p[1]=np.minimum(p[1],p[0]);p[2]=np.minimum(p[2],p[1])
    labels=np.zeros(p.shape[1:],np.uint8)
    for h in range(3):labels[p[h]>=.5]=h+1
    return labels


def train(pairs_path,output,head,*,epochs=30,device='cuda:0',cache_cases=4):
    pairs=read_json(pairs_path)
    if pairs.get('schema')!='sam_fullvolume_pairs_v1' or pairs['split']!='train':
        raise ValueError('Only supervised train320 SAM pairs may train this model')
    check_file(pairs['source_manifest']);check_file(pairs['collection'])
    source=manifest(pairs['source_manifest']['path'],allowed=('train',))
    allowed={r['case_id'] for r in source['cases']}
    if any(r['case_id'] not in allowed for r in pairs['cases']):
        raise ValueError('Foreign or validation/test case in SAM training pairs')
    records=[r for r in pairs['cases'] if r['head']==head]
    if not records or epochs<1 or cache_cases<0: raise ValueError('No records or invalid training settings')
    out=Path(output)
    if out.exists(): raise FileExistsError('Use a new output directory for full-volume adapter training')
    out.mkdir(parents=True)
    torch.set_num_threads(4)
    model=_model(head,device)
    # Checkpoint the original encoder without altering its parameters/keys.
    enc_forward=model.enc.forward
    model.enc.forward=lambda x:recompute(enc_forward,x,use_reentrant=False)
    optimizer=torch.optim.Adam(model.parameters(),lr=3e-3)
    schedule=torch.optim.lr_scheduler.MultiStepLR(optimizer,[15,25],gamma=.3)
    cuda=str(device).startswith('cuda')
    scaler=torch.amp.GradScaler('cuda',enabled=cuda,init_scale=1024.)
    rng=np.random.default_rng(20261118+head)
    cache=OrderedDict();history=[];best_score=-1.;best_state=None;best_epoch=None
    def get_items(row):
        key=row['case_id']
        if key in cache:
            value=cache.pop(key);cache[key]=value
            return value
        value=features(row,out.parent/'sdf_cache')
        if cache_cases:
            cache[key]=value
            while len(cache)>cache_cases:cache.popitem(last=False)
        return value
    torch.backends.cudnn.benchmark=False
    shapes={}
    for epoch in range(epochs):
        model.train();losses=[];updates=0
        for index in rng.permutation(len(records)):
            row=records[int(index)];key=row['case_id'];items=get_items(row)
            image,sam_soft,base_soft,gt,target,valid=[t.to(device) for t in items]
            shapes[key]=list(image.shape[-3:])
            optimizer.zero_grad(set_to_none=True)
            amp=torch.autocast('cuda',dtype=torch.float16) if cuda else nullcontext()
            with amp:
                pred,aux=model(image,sam_soft,base_soft)
            # Full-volume reductions and BCE remain FP32; AMP BCE is unsafe.
            aux={k:v.float() for k,v in aux.items()}
            loss=atlas._upstream()['fusion_loss'](pred.float(),gt,aux,target,valid)
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite full-volume Atlas loss')
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
            previous_scale=scaler.get_scale()
            scaler.step(optimizer);scaler.update()
            updates+=int(scaler.get_scale()>=previous_scale)
            losses.append(float(loss.detach()))
            del image,sam_soft,base_soft,gt,target,valid,pred,aux,loss
        if updates:schedule.step()
        model.eval();scores=[];baselines=[]
        with torch.no_grad():
            for row in records:
                image,sam_soft,base_soft,gt,_,_=get_items(row)
                truth=gt[0,0].numpy()>0
                base=base_soft[0,0].numpy()>=.5
                amp=torch.autocast('cuda',dtype=torch.float16) if cuda else nullcontext()
                with amp:pred,_=model(image.to(device),sam_soft.to(device),base_soft.to(device))
                output=pred[0,0].float().cpu().numpy()>=.5
                dice=lambda a:100.*(2*np.count_nonzero(a&truth)+1e-6)/(np.count_nonzero(a)+np.count_nonzero(truth)+1e-6)
                scores.append(dice(output));baselines.append(dice(base))
                del pred,image,sam_soft,base_soft,gt
        score=float(np.mean(scores))
        if score>best_score:
            best_score=score;best_epoch=epoch+1
            best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        history.append({'head':head,'epoch':epoch+1,'loss':float(np.mean(losses)),
                        'whole_volume_updates':len(losses),'train_dice_pct':score,
                        'optimizer_steps':updates,'amp_overflow_skips':len(losses)-updates,
                        'train_baseline_pct':float(np.mean(baselines)),
                        'train_gain_pp':score-float(np.mean(baselines))})
        print(history[-1],flush=True)
    del model.enc.forward
    metadata={'schema':'sam_fullvolume_atlas_v1','head':head,'epochs':epochs,'base_ch':16,
              'pairs':file_identity(pairs_path),'sam_contract':pairs['sam_contract'],'variant':pairs['variant'],
              'case_ids':sorted(shapes),'native_training_shapes':shapes,'resized':False,'cropped':False,
              'amp':cuda,'gradient_checkpointing':True,'cpu_cache_cases':cache_cases,
              'sdf_disk_cache':'lossless float32, keyed by input hashes; no spatial resize',
              'selected_epoch':best_epoch,'selected_train_dice_pct':best_score,
              'selection':'single global checkpoint per head by training Dice; not patient-specific oracle',
              'deployment_hierarchy':'WT preserved; TC=min(TC,WT); ET=min(ET,TC)',
              'atlas_source_sha256':atlas._sha(atlas.DEFAULT_SOURCE),
              'implementation_sha256':atlas._sha(__file__),'seed':20261118+head}
    path=out/'checkpoint.pth'
    torch.save({'metadata':metadata,'state_dict':best_state},path)
    write_json(out/'metadata.json',metadata);write_json(out/'history.json',history)
    return path


@torch.inference_mode()
def predict(checkpoint,mri,baseline,sam_binary,head,*,device='cuda:0'):
    saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
    meta=saved['metadata']
    if (meta.get('schema')!='sam_fullvolume_atlas_v1' or meta['head']!=head
            or meta['implementation_sha256']!=atlas._sha(__file__)
            or meta['atlas_source_sha256']!=atlas._sha(atlas.DEFAULT_SOURCE)):
        raise ValueError('Full-volume SAM fusion checkpoint/source mismatch')
    base=np.asarray(baseline)>head;sam=np.asarray(sam_binary,dtype=bool)
    if base.shape!=sam.shape:raise ValueError('Native SAM and baseline grids differ')
    if np.array_equal(base,sam):
        # Also defines the no-SAM ablation: identical inputs cannot change labels.
        return np.where(base,.99,.01).astype(np.float32)
    inputs=atlas.make_features(mri,base,sam)
    model=_model(head,device);model.load_state_dict(saved['state_dict'],strict=True)
    model.eval().requires_grad_(False)
    amp=torch.autocast('cuda',dtype=torch.float16) if str(device).startswith('cuda') else nullcontext()
    with amp:probability,_=model(*(t.to(device) for t in inputs))
    result=probability[0,0].float().cpu().numpy()
    if np.any((result>=.5)[base==sam]!=base[base==sam]):
        raise AssertionError('Two-source gate changed an agreeing source label')
    return result
