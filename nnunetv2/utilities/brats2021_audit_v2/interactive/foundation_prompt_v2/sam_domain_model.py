"""MRI adaptation of the *pretrained official SAM3* interactive decoder.

Only the original SAM prompt/mask decoder is optimized. The full 1008 x 1008
vision encoder remains frozen and may be cached. This is a BraTS adaptation,
not a reproduction of SAMRefiner++ (which learns IoU ranking LoRA).

No public prediction wrapper is used in the differentiable path: upstream
``predict_inst``/``_predict`` are inference helpers decorated with no_grad.
The trainable path calls the original PromptEncoder and MaskDecoder directly.
"""
from __future__ import annotations

import ast
from contextlib import nullcontext
from functools import lru_cache
import hashlib
import importlib
from pathlib import Path
import sys

import numpy as np
from scipy.ndimage import distance_transform_edt
import torch
from torch import nn
from torch.nn import functional as F

from .sam3_candidates import _load_official
from .sam_dense import prompt_geometry, SAM_SOURCE, SAM_CHECKPOINT

FEATURE_KEYS = ('image_embed', 'high_res_0', 'high_res_1')
SCHEMA = 'sam3_domain_decoder_v1'


def _source_imports(source):
    root = Path(source).expanduser().resolve()
    loaded = sys.modules.get('sam3')
    if loaded is not None and not Path(loaded.__file__).resolve().is_relative_to(root):
        raise RuntimeError('A different SAM3 source is already imported')
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    names = ('sam3.sam.prompt_encoder', 'sam3.sam.mask_decoder', 'sam3.sam.transformer')
    modules = [importlib.import_module(name) for name in names]
    if any(not Path(m.__file__).resolve().is_relative_to(root) for m in modules):
        raise RuntimeError('SAM3 class/source mismatch')
    return modules


def source_identity(source=SAM_SOURCE):
    paths = ('sam3/sam/prompt_encoder.py', 'sam3/sam/mask_decoder.py',
             'sam3/sam/transformer.py', 'sam3/model/sam3_tracker_base.py',
             'sam3/train/loss/loss_fns.py')
    return {p: hashlib.sha256((Path(source)/p).read_bytes()).hexdigest() for p in paths}


class FrozenSamDomainEncoder:
    """Official full-resolution encoder; returns CPU, ordinary FP16 tensors.

    High-resolution FPN levels have already passed through the pretrained
    decoder conv_s0/conv_s1. Those projections MUST remain frozen in the
    trainable decoder, otherwise cached features would become stale.
    """
    def __init__(self, source=SAM_SOURCE, checkpoint=SAM_CHECKPOINT, device='cuda:0'):
        self.source = str(Path(source).resolve())
        self.checkpoint = str(Path(checkpoint).resolve())
        self.device = str(device)
        self.model, self.processor = _load_official(self.source, self.checkpoint, self.device)
        if self.processor.resolution != 1008:
            raise ValueError('The pretrained SAM3 encoder must keep resolution 1008')
        self.model.eval().requires_grad_(False)

    def encode(self, image):
        from PIL import Image
        if isinstance(image, np.ndarray):
            if image.ndim != 3 or image.shape[-1] != 3 or image.dtype != np.uint8:
                raise ValueError('Expected a whole native RGB uint8 slice [H,W,3]')
            image = Image.fromarray(image)
        if not isinstance(image, Image.Image) or image.mode != 'RGB':
            raise ValueError('Expected RGB PIL image or RGB uint8 array')
        amp = torch.autocast('cuda', dtype=torch.float16) if self.device.startswith('cuda') else nullcontext()
        with torch.no_grad(), amp:
            state = self.processor.set_image(image)
            fpn = state['backbone_out']['sam2_backbone_out']['backbone_fpn']
            if len(fpn) != 3:
                raise ValueError('Unexpected SAM3 FPN levels')
            tracker = self.model.inst_interactive_predictor.model
            # This is exactly the no-memory embedding addition in predict_inst.
            embedding = fpn[2] + tracker.no_mem_embed.reshape(1, -1, 1, 1)
            tensors = (embedding, fpn[0], fpn[1])
            # Clone outside inference_mode: inference tensors cannot be saved
            # for backward by decoder convolutions even when requires_grad=False.
        return {name: value.detach().to(device='cpu', dtype=torch.float16).clone()
                for name, value in zip(FEATURE_KEYS, tensors)}


class LoRALinear(nn.Module):
    """Optional Q/V low-rank update; an explicit adaptation, not upstream code."""
    def __init__(self, original, rank=4, alpha=4.):
        super().__init__()
        if not isinstance(original, nn.Linear) or rank < 1:
            raise ValueError('LoRA needs a Linear and a positive rank')
        self.base = original.requires_grad_(False)
        self.down = nn.Linear(original.in_features, rank, bias=False)
        self.up = nn.Linear(rank, original.out_features, bias=False)
        nn.init.kaiming_uniform_(self.down.weight, a=5**.5)
        nn.init.zeros_(self.up.weight)
        self.scale = float(alpha)/rank

    def forward(self, x):
        return self.base(x) + self.scale*self.up(self.down(x))


class SamDomainDecoder(nn.Module):
    def __init__(self, source=SAM_SOURCE, *, mode='decoder', lora_rank=4,
                 checkpoint_decoder=False):
        super().__init__()
        if mode not in ('decoder', 'qv_lora'):
            raise ValueError('mode must be decoder or qv_lora')
        prompt, decoder, transformer = _source_imports(source)
        self.source = str(Path(source).resolve())
        self.mode, self.lora_rank = mode, int(lora_rank)
        self.checkpoint_decoder = bool(checkpoint_decoder)
        self.sam_prompt_encoder = prompt.PromptEncoder(
            embed_dim=256, image_embedding_size=(72,72),
            input_image_size=(1008,1008), mask_in_chans=16)
        self.sam_mask_decoder = decoder.MaskDecoder(
            num_multimask_outputs=3,
            transformer=transformer.TwoWayTransformer(
                depth=2, embedding_dim=256, mlp_dim=2048, num_heads=8),
            transformer_dim=256, iou_head_depth=3, iou_head_hidden_dim=256,
            use_high_res_features=True, iou_prediction_use_sigmoid=True,
            pred_obj_scores=True, pred_obj_scores_mlp=True,
            use_multimask_token_for_obj_ptr=True,
            # A single supervised token is used in BOTH train and inference.
            # Dynamic stability fallback would change the model at evaluation.
            dynamic_multimask_via_stability=False)
        self._adaptation_installed = False

    def install_trainable_mode(self):
        if self._adaptation_installed:
            return self
        self.requires_grad_(self.mode == 'decoder')
        if self.mode == 'qv_lora':
            for module in list(self.sam_mask_decoder.transformer.modules()):
                for name in ('q_proj','v_proj'):
                    layer = getattr(module, name, None)
                    if isinstance(layer, nn.Linear):
                        setattr(module, name, LoRALinear(layer, self.lora_rank, self.lora_rank))
            # Prompt mask embedding adapts to MRI masks as well.
            self.sam_prompt_encoder.mask_downscaling.requires_grad_(True)
        # Cached high-res levels already include these projections.
        for name in ('conv_s0','conv_s1','iou_prediction_head','pred_obj_score_head'):
            getattr(self.sam_mask_decoder,name).requires_grad_(False)
        for head in self.sam_mask_decoder.output_hypernetworks_mlps[1:]:
            head.requires_grad_(False)
        # A dense mask prompt is always supplied by this method.
        self.sam_prompt_encoder.no_mask_embed.requires_grad_(False)
        self._adaptation_installed = True
        return self

    def configuration(self):
        return {'schema':SCHEMA,'source':self.source,'mode':self.mode,
                'lora_rank':self.lora_rank,'checkpoint_decoder':self.checkpoint_decoder,
                'encoder_resolution':1008,'native_slice_not_cropped':True,
                'features_after_frozen_highres_projection':True,
                'dynamic_multimask_via_stability':False,
                'official_source_sha256':source_identity(self.source)}

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def mask_input_size(self):
        return self.sam_prompt_encoder.mask_input_size

    def prompts_from_mask(self, baseline):
        binary=np.asarray(baseline,dtype=bool)
        if binary.ndim != 2:
            raise ValueError('Expected one full native 2D baseline mask')
        if binary.any():
            points,box=prompt_geometry(binary)
            point_labels=np.ones((1,len(points)),np.int64)
            points=points[None]
            sdf=distance_transform_edt(binary)-distance_transform_edt(~binary)
        else:
            # Deterministic no-GT empty-mask encoding, never a GT-derived point.
            points=np.zeros((1,1,2),np.float32)
            point_labels=-np.ones((1,1),np.int64)
            box=np.array([0,0,binary.shape[1]-1,binary.shape[0]-1],np.float32)
            sdf=np.full(binary.shape,-12.,np.float32)
        dense=torch.from_numpy(np.clip(sdf/2.,-6.,6.).astype(np.float32))[None,None]
        dense=F.interpolate(dense,size=self.mask_input_size,mode='bilinear',align_corners=False)
        return {'points':torch.from_numpy(points),'point_labels':torch.from_numpy(point_labels),
                'box':torch.from_numpy(box[None]),'mask_input':dense}

    def forward(self, features, baseline):
        return self.forward_cached(features,self.prompts_from_mask(baseline),np.shape(baseline))

    def forward_cached(self, features, prompts, original_size):
        """Differentiable official decoder forward; no GT or target argument.

        Features retain batch axis; native prompts have points [B,N,2] XY,
        labels [B,N], box [B,4] XYXY, mask_input [B,1,288,288]. Output [B,1,H,W].
        """
        h,w=map(int,original_size)
        if min(h,w)<1 or set(features)!=set(FEATURE_KEYS):
            raise ValueError('Invalid native size or cached feature keys')
        items={k:torch.as_tensor(v,device=self.device,dtype=torch.float32) for k,v in features.items()}
        expected={'image_embed':(256,72,72),'high_res_0':(32,288,288),'high_res_1':(64,144,144)}
        if any(v.ndim!=4 or tuple(v.shape[1:])!=expected[k] for k,v in items.items()):
            raise ValueError('Cached SAM features do not preserve official encoder resolution')
        batch=items['image_embed'].shape[0]
        if any(v.shape[0]!=batch for v in items.values()):
            raise ValueError('Cached feature batch sizes differ')
        points=torch.as_tensor(prompts['points'],device=self.device,dtype=torch.float32).clone()
        labels=torch.as_tensor(prompts['point_labels'],device=self.device,dtype=torch.int64)
        boxes=torch.as_tensor(prompts['box'],device=self.device,dtype=torch.float32).reshape(batch,2,2).clone()
        dense=torch.as_tensor(prompts['mask_input'],device=self.device,dtype=torch.float32)
        if (points.ndim!=3 or points.shape[0]!=batch or points.shape[-1]!=2
                or labels.shape!=points.shape[:2] or dense.shape!=(batch,1,*self.mask_input_size)):
            raise ValueError('Invalid native prompt shapes')
        # Exact SAM2Transforms normalization from native XY to 1008 image grid.
        scale=points.new_tensor([1008./w,1008./h])
        points=points*scale; boxes=boxes*scale
        merged_points=torch.cat([boxes,points],dim=1)
        box_labels=labels.new_tensor([2,3])[None].expand(batch,-1)
        merged_labels=torch.cat([box_labels,labels],dim=1)
        sparse,dense_embeddings=self.sam_prompt_encoder(
            points=(merged_points,merged_labels),boxes=None,masks=dense)
        def decode(image_embed,high0,high1,sparse,dense_embeddings):
            logits,_,_,_=self.sam_mask_decoder(
                image_embeddings=image_embed,
                image_pe=self.sam_prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse,dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,repeat_image=False,high_res_features=[high0,high1])
            return logits
        arguments=(items['image_embed'],items['high_res_0'],items['high_res_1'],sparse,dense_embeddings)
        if self.training and self.checkpoint_decoder:
            from torch.utils.checkpoint import checkpoint
            logits=checkpoint(decode,*arguments,use_reentrant=False)
        else:
            logits=decode(*arguments)
        # No hard morphology or discrete mask operations in the gradient path.
        return F.interpolate(logits.float(),size=(h,w),mode='bilinear',align_corners=False)


def build_domain_decoder(source=SAM_SOURCE, checkpoint=None, *, mode='decoder',
                         lora_rank=4, checkpoint_decoder=False, configuration=None,
                         device='cpu'):
    """Build only original pretrained heads, without allocating the giant backbone.

    ``checkpoint`` is the original SAM3 checkpoint for initial training. For a
    trained checkpoint, pass its configuration then load its state_dict strictly.
    """
    if configuration is None and checkpoint is None:
        # New training always starts from real SAM weights, never a silently
        # random decoder. Restoration supplies a configuration + strict state.
        checkpoint=SAM_CHECKPOINT
    if configuration is not None:
        if configuration.get('schema')!=SCHEMA:
            raise ValueError('Unsupported SAM domain decoder configuration')
        source=configuration['source'];mode=configuration['mode']
        lora_rank=configuration['lora_rank'];checkpoint_decoder=configuration['checkpoint_decoder']
        if configuration['official_source_sha256']!=source_identity(source):
            raise ValueError('Official SAM3 source changed since checkpoint creation')
    model=SamDomainDecoder(source,mode=mode,lora_rank=lora_rank,checkpoint_decoder=checkpoint_decoder)
    if checkpoint is not None:
        state=torch.load(checkpoint,map_location='cpu',weights_only=True,mmap=True)
        if 'model' in state: state=state['model']
        prefix='tracker.'
        selected={k[len(prefix):]:v for k,v in state.items()
                  if k.startswith(prefix+'sam_prompt_encoder.') or k.startswith(prefix+'sam_mask_decoder.')}
        if not selected:
            raise ValueError('No pretrained tracker prompt/mask decoder keys in SAM3 checkpoint')
        model.load_state_dict(selected,strict=True)
        del selected,state
    model.install_trainable_mode()
    return model.to(device)


@lru_cache(maxsize=4)
def _official_losses(source):
    """Reuse exact upstream loss definitions without optional torchmetrics/Triton imports."""
    path=Path(source)/'sam3/train/loss/loss_fns.py'
    parsed=ast.parse(path.read_text())
    wanted={'_dice_loss','sigmoid_focal_loss'}
    definitions=[node for node in parsed.body if isinstance(node,ast.FunctionDef) and node.name in wanted]
    if {n.name for n in definitions}!=wanted:
        raise ValueError('Official SAM3 loss functions changed')
    module=ast.Module(body=definitions,type_ignores=[])
    namespace={'torch':torch,'F':F}
    exec(compile(ast.fix_missing_locations(module),str(path),'exec'),namespace)
    return namespace


def sam_supervised_loss(logits, target, *, source=SAM_SOURCE, focal_weight=20., dice_weight=1.):
    """Original SAM3 focal and Dice; targets enter only this supervised loss.

    Flatten to [B,HW] for upstream focal's spatial mean, and force its exact
    PyTorch implementation (triton=False). All reductions are FP32.
    """
    targets=torch.as_tensor(target,device=logits.device,dtype=torch.float32)
    if targets.shape!=logits.shape:
        raise ValueError('Native prediction and supervised target shapes must match')
    functions=_official_losses(str(Path(source).resolve()))
    prediction=logits.float()
    dice=functions['_dice_loss'](prediction.flatten(1),targets.flatten(1),len(prediction))
    focal=functions['sigmoid_focal_loss'](prediction.flatten(1),targets.flatten(1),len(prediction),triton=False)
    total=float(focal_weight)*focal+float(dice_weight)*dice
    return total,{'dice':dice.detach(),'focal':focal.detach()}
