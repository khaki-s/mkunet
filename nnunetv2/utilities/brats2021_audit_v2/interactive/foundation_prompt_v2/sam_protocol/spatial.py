"""Full-native 3D fusion of fixed-prompt SAM and nnInteractive evidence."""
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from ..common import check_file, file_identity, read_json, save_mask, write_json
from . import candidates, dense, residual
from .cohorts import rows

VARIANT = 'joint_spatial'
TRAINER = 'gpu_spatial'
SCHEMA = 'brats_native_spatial_fusion_v1'
CHANNELS = 4 + 1 + 2 + 6


class SpatialReadout(nn.Module):
    """Dense original-resolution output, with multiscale internal feature maps."""

    def __init__(self):
        super().__init__()
        self.fine = nn.Conv3d(CHANNELS, 1, 1)
        self.down = nn.Sequential(nn.Conv3d(CHANNELS, 12, 3, stride=2, padding=1),
                                  nn.GroupNorm(3, 12), nn.SiLU())
        self.context = nn.Sequential(nn.Conv3d(12, 16, 3, stride=2, padding=1),
                                     nn.GroupNorm(4, 16), nn.SiLU(),
                                     nn.Conv3d(16, 16, 3, dilation=2, padding=2),
                                     nn.GroupNorm(4, 16), nn.SiLU())
        self.readout = nn.Sequential(nn.Conv3d(28, 12, 3, padding=1),
                                     nn.GroupNorm(3, 12), nn.SiLU(), nn.Conv3d(12, 1, 1))

    def forward(self, x):
        shallow = self.down(x)
        deep = self.context(shallow)
        deep = F.interpolate(deep, size=shallow.shape[-3:], mode='trilinear', align_corners=False)
        small = self.readout(torch.cat((shallow, deep), dim=1))
        return self.fine(x).float() + F.interpolate(small.float(), size=x.shape[-3:],
                                                   mode='trilinear', align_corners=False)


def _mri_scaling(mri):
    scaled = np.zeros(mri.shape, dtype=np.float16)
    for h,image in enumerate(mri):
        probe = image[::2, ::2, ::2]
        voxels = probe[probe != 0]
        if not len(voxels):
            continue
        q10, q50, q90 = np.percentile(voxels, (10, 50, 90))
        use = image != 0
        scaled[h,use] = np.clip((image[use]-q50)/max(1e-6,q90-q10), -6, 6).astype(np.float16)
    return scaled


def _write_npy(path, value):
    temp = path.with_suffix('.tmp.npy')
    np.save(temp, value, allow_pickle=False)
    temp.replace(path)
    return file_identity(path)


def _source_files():
    return {name:file_identity(path)['sha256'] for name,path in (
        ('spatial', __file__), ('dense', dense.__file__), ('candidates', candidates.__file__))}


def _freeze_volumes(selected, raw_dir, responses_dir, output, *, labeled):
    root = Path(output)
    method = _source_files()
    for row in selected:
        folder = root/row['case_id']
        clean = {k:v for k,v in row.items() if k != 'gt'}
        sam_index = read_json(Path(raw_dir)/row['case_id']/'sam.json')
        responses_index = read_json(Path(responses_dir)/row['case_id']/'responses.json')
        if ('gt' in row) != labeled:
            raise ValueError('Only training volumes may include ground truth')
        if (sam_index['row'] != clean or responses_index['row'] != clean
                or sam_index['variants'] != list(dense.VARIANTS)
                or responses_index['kinds'] != list(candidates.RESPONSE_NAMES)
                or responses_index.get('gt_used') is not False
                or responses_index['sam'] != sam_index['arrays']):
            raise ValueError('Native spatial volume requires matching label-free SAM and nnInteractive receipts')
        for identity in (sam_index['arrays'], responses_index['arrays']):
            check_file(identity)
        path = folder/'volume.json'
        binding = {'row':row, 'sam':sam_index['arrays'], 'responses':responses_index['arrays'],
                   'method':method}
        if path.exists():
            cached = read_json(path)
            if cached['binding'] != binding or cached['labeled'] != labeled:
                raise ValueError('Native volume belongs to a different training record')
            for identity in cached['volumes'].values():
                check_file(identity)
            continue
        mri, base, ref = dense.load_native(clean)
        folder.mkdir(parents=True, exist_ok=True)
        image = _mri_scaling(mri)
        volumes = {'mri':_write_npy(folder/'mri.npy', image),
                   'baseline':_write_npy(folder/'baseline.npy', base),
                   'brain':_write_npy(folder/'brain.npy', np.any(mri != 0, axis=0))}
        if labeled:
            volumes['truth'] = _write_npy(folder/'truth.npy', residual._truth(row, ref))
        write_json(path, {'binding':binding, 'volumes':volumes, 'labeled':labeled,
                          'shape':list(base.shape), 'affine':ref.affine.tolist()})
        print(f'Native spatial volumes {row["case_id"]} labeled={labeled}', flush=True)


def prepare_spatial(args):
    labeled = args.split == 'train'
    selected = rows(args.cohort, args.split, truth=labeled)[args.shard_index::args.num_shards]
    _freeze_volumes(selected,args.dense,args.responses,args.output,labeled=labeled)
    return args.output


def _index(selected, folder, *, labeled):
    result = {}
    method = _source_files()
    for row in selected:
        path = Path(folder)/row['case_id']/'volume.json'
        record = read_json(path)
        if (record['binding']['row'] != row or record['labeled'] != labeled
                or record['binding']['method'] != method):
            raise ValueError('Full native volume case or code mismatch')
        for identity in record['volumes'].values():
            check_file(identity)
        for key in ('sam', 'responses'):
            check_file(record['binding'][key])
        result[row['case_id']] = record
    return result


def _unpack(packed_path, shape, head):
    with np.load(packed_path,allow_pickle=False) as data:
        packed = data['packed'][:,head,...]
    return np.unpackbits(packed,axis=-1,count=int(np.prod(shape))).reshape((6,*shape)).astype(bool,copy=False)


def _input(record, head, device, *, labeled):
    shape = tuple(record['shape'])
    access = record['volumes']
    mri = np.load(access['mri']['path'],mmap_mode='r',allow_pickle=False)
    base = np.load(access['baseline']['path'],mmap_mode='r',allow_pickle=False)>head
    brain = np.load(access['brain']['path'],mmap_mode='r',allow_pickle=False)
    source = record['binding']
    with np.load(source['sam']['path'],allow_pickle=False) as a:
        sam = a['masks'][:,head]
    response = _unpack(source['responses']['path'],shape,head)
    support = brain & np.any(response != base,axis=0)
    x = np.empty((CHANNELS,*shape),np.float16)
    x[:4], x[4], x[5:7], x[7:] = mri, base, sam, response
    x = torch.as_tensor(x,device=device).unsqueeze(0).contiguous(memory_format=torch.channels_last_3d)
    selected = np.flatnonzero(support)
    target = None
    if labeled:
        if not record['labeled']:
            raise ValueError('Training loss requires an explicit training record')
        target = np.load(access['truth']['path'],mmap_mode='r',allow_pickle=False)>head
    return x, base, selected, target


def _loss(scores, base, target, eligible, device):
    values = scores.flatten().index_select(0, torch.as_tensor(eligible,device=device))
    y = torch.as_tensor(target.ravel()[eligible],dtype=torch.float32,device=device)
    old = torch.as_tensor(base.ravel()[eligible],dtype=torch.float32,device=device)
    weight = 1. + (y != old).float()
    bce = F.binary_cross_entropy_with_logits(values,y,weight=weight)
    prob = values.sigmoid()
    old_count = int(base.sum()) - int(old.sum().item())
    fixed_tp = int(np.count_nonzero(base&target)) - int((old*y).sum().item())
    dice = (2.*(fixed_tp+(prob*y).sum())+1.)/(old_count+int(target.sum())+prob.sum()+1.)
    return bce+.35*(1.-dice)


def _model(device):
    torch.manual_seed(61119)
    return SpatialReadout().to(device=device, memory_format=torch.channels_last_3d)


def _train(records, patients, h, epochs, device):
    network = _model(device)
    optimizer = torch.optim.AdamW(network.parameters(),lr=.002,weight_decay=.01)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=epochs,eta_min=.00015)
    scaler = torch.amp.GradScaler('cuda',init_scale=512.)
    generator = np.random.default_rng(8830+h)
    for epoch in range(epochs):
        network.train()
        losses=[]
        for i in generator.permutation(len(patients)):
            key=patients[int(i)]
            x, base, support, target = _input(records[key],h,device,labeled=True)
            if len(support)==0:
                continue
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.float16):
                pred=network(x)
            loss=_loss(pred,base,target,support,device)
            if not torch.isfinite(loss):
                raise RuntimeError(f'Nonfinite native loss for {key}')
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(network.parameters(),4.)
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach()))
        schedule.step()
        print(f'native spatial head={h} n={len(patients)} epoch={epoch+1}/{epochs} loss={float(np.mean(losses)):.6f}',flush=True)
    return network.eval()


@torch.inference_mode()
def _prob(network,x,support,device):
    if len(support)==0:
        return np.empty(0,dtype=np.float32)
    with torch.autocast('cuda',dtype=torch.float16):
        logits=network(x)
    return logits.flatten().index_select(0,torch.as_tensor(support,device=device)).float().sigmoid().cpu().numpy()


def fit_spatial(args):
    device=torch.device(args.device)
    if device.type!='cuda' or args.epochs<1:
        raise ValueError('Spatial fitting requires native CUDA execution and at least one epoch')
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    cohort=rows(args.cohort,'train',truth=True)
    originals=_index(cohort,args.volumes,labeled=True)
    ids=list(originals)
    check=[k for i,k in enumerate(ids) if i%5==4]
    fit=[k for i,k in enumerate(ids) if i%5!=4]
    h=args.head
    destination=Path(args.output)/f'{dense.HEADS[h]}.pth'
    contract={'schema':SCHEMA,'cohort':file_identity(args.cohort),'head':h,'train_ids':ids,
              'calibration_ids':check,'method':_source_files(),'epochs':args.epochs,'trainer':TRAINER}
    if destination.exists():
        if torch.load(destination,map_location='cpu',weights_only=True)['binding']!=contract:
            raise ValueError('Existing model uses different data or source')
        return destination
    provisional=_train(originals,fit,h,args.epochs,device)
    scores=[]
    for key in check:
        x,base,support,target=_input(originals[key],h,device,labeled=True)
        p=_prob(provisional,x,support,device)
        scores.append(residual._gains(p,support,base,target))
    del provisional
    scores=np.stack(scores)
    objective=scores.mean(axis=0)-.18*np.maximum(0.,-scores).mean(axis=0)
    options=(*residual.THRESHOLDS,None)
    i,j=np.unravel_index(np.argmax(objective),objective.shape) if objective.max()>0 else (len(options)-1,)*2
    calibration={'add':options[i],'remove':options[j],'calibration_gain_pp':float(scores[:,i,j].mean()),
                 'case_gains_pp':scores[:,i,j].tolist()}
    model=_train(originals,ids,h,args.epochs,device)
    destination.parent.mkdir(parents=True,exist_ok=True)
    temporary=destination.with_suffix('.tmp')
    torch.save({'binding':contract,'calibration':calibration,'model':model.cpu().state_dict()},temporary)
    temporary.replace(destination)
    write_json(destination.with_suffix('.json'),{'binding':contract,'calibration':calibration,
        'peak_allocated_gib':torch.cuda.max_memory_allocated(device)/2**30,
        'peak_reserved_gib':torch.cuda.max_memory_reserved(device)/2**30})
    return destination


def load_models(root,cohort):
    paths=[Path(root)/f'{h}.pth' for h in dense.HEADS]
    states=[torch.load(p,map_location='cpu',weights_only=True) for p in paths]
    identity=file_identity(cohort)
    for h,state in enumerate(states):
        version=state['binding']
        if (version['schema']!=SCHEMA or version['cohort']!=identity or version['head']!=h
                or version['method']!=_source_files()):
            raise ValueError('Native fusion checkpoint binding changed')
    return states,[file_identity(p) for p in paths]


def infer_spatial(selected, model_root, cohort, volumes, output, device):
    states,ids=load_models(model_root,cohort)
    models=[_model(device) for _ in states]
    for network,state in zip(models,states):
        network.load_state_dict(state['model'])
        network.eval()
    features=_index(selected,volumes,labeled=False)
    for row in selected:
        key=row['case_id']
        record=features[key]
        folder=Path(output)/key
        binding={'row':row,'models':ids,'selected':VARIANT,'variants':[VARIANT],
                 'sam':record['binding']['sam'],'responses':record['binding']['responses']}
        receipt=folder/'prediction.json'
        if receipt.exists():
            previous=read_json(receipt)
            if previous['binding']!=binding:
                raise ValueError('Existing prediction has another model')
            check_file(previous['result'][VARIANT]['mask'])
            continue
        outputs,edits=[],[]
        for h,(network,state) in enumerate(zip(models,states)):
            x,base,coords,_=_input(record,h,device,labeled=False)
            p=_prob(network,x,coords,device)
            limits=state['calibration']
            add,remove=residual._changed(p,coords,base,limits['add'],limits['remove'])
            new=np.ascontiguousarray(base).copy()
            new.ravel()[coords[add]]=True
            new.ravel()[coords[remove]]=False
            outputs.append(new)
            edits.append({'head':dense.HEADS[h],'available':len(coords),'added':int(add.sum()),'removed':int(remove.sum())})
        reference=nib.load(row['prediction']['path'])
        path=folder/f'{VARIANT}.nii.gz'
        identity=save_mask(path,residual.merge_heads(outputs),reference)
        write_json(receipt,{'binding':binding,'gt_used':False,'result':{VARIANT:{'mask':identity,'edits':edits}}})
        print(f'Native spatial predict {key} mask={row["mask_code"]}',flush=True)
    return output


def predict_spatial(args):
    selected=rows(args.cohort,'validation')[args.shard_index::args.num_shards]
    device=torch.device(args.device)
    torch.cuda.set_device(device)
    return infer_spatial(selected,args.model,args.cohort,args.volumes,args.output,device)
