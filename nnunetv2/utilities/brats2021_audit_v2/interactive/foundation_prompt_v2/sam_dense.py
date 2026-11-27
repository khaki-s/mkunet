"""Full-slice SAM3 mask refinement. Prompts depend only on the initial mask."""
from contextlib import nullcontext

import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt, label
import torch
from torch.nn import functional as F

from .sam3_candidates import _load_official

HEADS = ('WT', 'TC', 'ET')
MODALITIES = {'WT': 3, 'TC': 1, 'ET': 1}
VARIANTS = ('mask_box_point', 'box_point')
SAM_SOURCE = '/data02/chujingyi/sam3-main'
SAM_CHECKPOINT = '/data02/chujingyi/model/sam3/master/sam3.pt'


def prompt_geometry(mask, *, padding=4, max_points=3):
    """Distance-centre points in separate predicted components and a native box."""
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2 or not mask.any():
        raise ValueError('A nonempty native 2D mask is required')
    coordinates = np.argwhere(mask)
    lo = np.maximum(coordinates.min(0)-padding, 0)
    hi = np.minimum(coordinates.max(0)+padding+1, mask.shape)
    components, _ = label(mask)
    sizes = np.bincount(components.ravel()); sizes[0] = 0
    points=[]
    for index in np.argsort(-sizes):
        if not sizes[index] or len(points)>=max_points: break
        component=components==index
        distance=distance_transform_edt(np.pad(component,1))[1:-1,1:-1]
        row,column=np.unravel_index(np.argmax(distance),mask.shape)
        points.append([float(column),float(row)])
    return np.array(points,np.float32), np.array([lo[1],lo[0],hi[1]-1,hi[0]-1],np.float32)


class SamDenseRefiner:
    def __init__(self, device='cuda:0', *, source=SAM_SOURCE, checkpoint=SAM_CHECKPOINT,
                 model=None, processor=None):
        self.device=str(device)
        if (model is None)!=(processor is None):
            raise ValueError('Supply both model and processor for injected tests')
        if model is None: model,processor=_load_official(source,checkpoint,self.device)
        self.model,self.processor=model,processor
        if getattr(processor,'resolution',1008)!=1008:
            raise ValueError('Keep the official SAM3 image encoder resolution of 1008')
        self.model.eval().requires_grad_(False)
        encoder=self.model.inst_interactive_predictor.model.sam_prompt_encoder
        self.mask_size=tuple(encoder.mask_input_size)

    @torch.inference_mode()
    def generate(self,mri,current):
        images=np.asarray(mri,dtype=np.float32); current=np.asarray(current)
        if (images.shape!=(4,*current.shape) or current.ndim!=3 or not np.isfinite(images).all()
                or not np.isin(current,(0,1,2,3)).all()):
            raise ValueError('Require finite MRI[4,X,Y,Z] and native internal labels[X,Y,Z]')
        outputs={variant:{head:(current>h).astype(np.uint8).copy() for h,head in enumerate(HEADS)}
                 for variant in VARIANTS}
        calls=0; images_encoded=0; slice_records=[]
        # Encode each modality/slice once, reuse its full-resolution feature state
        # for TC and ET and for both prompt alternatives. Never batch images.
        for modality in sorted(set(MODALITIES.values())):
            available=images[modality][images[modality]!=0]
            lower,upper=np.percentile(available,(1,99)) if len(available) else (0.,1.)
            span=max(float(upper-lower),1e-6)
            selected=[(h,head) for h,head in enumerate(HEADS) if MODALITIES[head]==modality]
            slices=np.flatnonzero(np.logical_or.reduce([(current>h).any(axis=(0,1)) for h,_ in selected]))
            for z in slices:
                gray=(np.clip((images[modality,:,:,z]-lower)/span,0,1)*255).astype(np.uint8)
                rgb=Image.fromarray(np.repeat(gray[...,None],3,axis=2))
                amp=torch.autocast('cuda',dtype=torch.float16) if self.device.startswith('cuda') else nullcontext()
                with amp:
                    state=self.processor.set_image(rgb)
                    images_encoded+=1
                    for h,head in selected:
                        mask=current[:,:,z]>h
                        if not mask.any(): continue
                        points,box=prompt_geometry(mask)
                        sdf=distance_transform_edt(mask)-distance_transform_edt(~mask)
                        logits=torch.from_numpy(np.clip(sdf/2.,-6.,6.).astype(np.float32))[None,None]
                        lowres=F.interpolate(logits,size=self.mask_size,mode='bilinear',align_corners=False)[0].numpy()
                        for variant in VARIANTS:
                            masks,scores,_=self.model.predict_inst(state,point_coords=points,
                                point_labels=np.ones(len(points),np.int32),box=box,
                                mask_input=lowres if variant=='mask_box_point' else None,
                                multimask_output=True,return_logits=False,normalize_coords=True)
                            masks=np.asarray(masks);scores=np.asarray(scores).reshape(-1)
                            if (masks.ndim!=3 or masks.shape[1:]!=mask.shape or len(masks)!=len(scores)
                                    or not np.isfinite(masks).all() or not np.isfinite(scores).any()):
                                raise ValueError('SAM3 returned invalid native masks/scores')
                            chosen=int(np.argmax(np.where(np.isfinite(scores),scores,-np.inf)))
                            outputs[variant][head][:,:,z]=masks[chosen]>0
                            calls+=1
                            slice_records.append({'head':head,'slice':int(z),'modality':modality,'variant':variant,
                                'quality_uncalibrated':float(scores[chosen]),'points':points.tolist(),
                                'box':box.tolist(),'changed_voxels':int(np.count_nonzero(mask!=(masks[chosen]>0)))})
                del state
            print(f'SAM3 modality={modality}, full native slices={len(slices)}, calls={calls}',flush=True)
        return outputs,{'sam_image_resolution':1008,'native_shape':list(current.shape),'mask_input_size':list(self.mask_size),
                        'image_encodes':images_encoded,'sam_predict_calls':calls,'gt_in_prompts':False,
                        'frozen_parameters':not any(p.requires_grad for p in self.model.parameters()),
                        'empty_initial_slices':'preserve initial empty slice','slice_records':slice_records}
