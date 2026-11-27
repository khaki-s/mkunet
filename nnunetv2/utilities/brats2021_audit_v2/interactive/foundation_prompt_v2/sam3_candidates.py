"""Frozen official SAM 3 proposals for a learned prompt-utility selector.

This is a BraTS adaptation, not a reproduction of PromptMoE/AtlasSegFM.
SAM 3 supplies image features and competing masks; disagreement is a *candidate*
correction, never a truth label. No GT is accepted by this module. A separate
utility router must learn whether applying a candidate through nnInteractive
helps. The geometry is deliberately anchored in the existing segmentation:
an entirely absent region cannot reliably be discovered by this implementation.
"""
from __future__ import annotations

import importlib
import importlib.util
import importlib.metadata
import contextlib
import io
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_edt, label
import torch
from torch.nn import functional as F

HEADS = ('WT', 'TC', 'ET')
MODALITIES = ('T1', 'T1ce', 'T2', 'FLAIR')
SLOT_NAMES = tuple(f'{head}{sign}' for head in HEADS for sign in ('+', '-'))
GLOBAL_CONTEXT_DIM = 4 * 256 + 3 * 4
SLOT_CONTEXT_DIM = 256 + 8
CONTEXT_DIM = GLOBAL_CONTEXT_DIM + 6 * SLOT_CONTEXT_DIM


def validate_source(source_dir, checkpoint, *, check_imports=True):
    """Inspect paths/dependencies only; does not allocate a model or open 3GB weights."""
    source = Path(source_dir).expanduser().resolve()
    weights = Path(checkpoint).expanduser().resolve()
    required = ('sam3/model_builder.py', 'sam3/model/sam3_image_processor.py',
                'sam3/model/sam1_task_predictor.py', 'sam3/assets/bpe_simple_vocab_16e6.txt.gz')
    missing_files = [str(source / path) for path in required if not (source / path).is_file()]
    if not weights.is_file() or weights.stat().st_size == 0:
        missing_files.append(str(weights))
    dependencies = ('torch', 'torchvision', 'PIL', 'timm', 'ftfy', 'iopath',
                    'huggingface_hub', 'einops', 'scipy', 'psutil', 'pycocotools',
                    'regex', 'pkg_resources', 'triton', 'typing_extensions', 'skimage',
                    'wcwidth', 'portalocker')
    missing_dependencies = []
    if check_imports:
        for package in dependencies:
            try:
                found = importlib.util.find_spec(package) is not None
            except (ImportError, ValueError):
                found = False
            if not found:
                missing_dependencies.append(package)
    versions = {}
    for distribution in ('torch', 'torchvision', 'numpy', 'setuptools', 'timm'):
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = None
    cautions = []
    if sys.version_info < (3, 12):
        cautions.append('Official SAM 3 README recommends Python >=3.12')
    if (versions['numpy'] or '').startswith('2.'):
        cautions.append('SAM 3 pyproject requires numpy>=1.26,<2; current NumPy is 2.x')
    if '+cu11' in (versions['torch'] or ''):
        cautions.append('Official README recommends CUDA >=12.6; installed PyTorch uses CUDA 11.x')
    return {'source_dir': str(source), 'checkpoint': str(weights),
            'checkpoint_bytes': weights.stat().st_size if weights.is_file() else None,
            'missing_files': missing_files, 'missing_dependencies': missing_dependencies,
            'ready_for_import': not missing_files and not missing_dependencies,
            'weight_load_tested': False, 'inference_tested': False,
            'python': sys.version.split()[0],
            'software_versions': versions, 'runtime_cautions': cautions,
            'note': 'Path/dependency check only; official README recommends Python >=3.12. '
                    'A successful check does not prove checkpoint compatibility or CUDA support.'}


def _load_official(source_dir, checkpoint, device):
    audit = validate_source(source_dir, checkpoint)
    if not audit['ready_for_import']:
        raise RuntimeError(f'SAM 3 preflight failed: {audit}. Install dependencies in a '
                           'dedicated SAM 3 environment; no fallback/random model is used.')
    source = str(Path(source_dir).resolve())
    # Refuse silently mixing an installed SAM version with this pinned local source.
    loaded = sys.modules.get('sam3')
    if loaded is not None and not Path(loaded.__file__).resolve().is_relative_to(source):
        raise RuntimeError('A different sam3 package is already imported; use a fresh process.')
    sys.path.insert(0, source)
    try:
        builder = importlib.import_module('sam3.model_builder')
        processor_module = importlib.import_module('sam3.model.sam3_image_processor')
    except ImportError as error:
        raise RuntimeError(f'Official SAM 3 import failed: {error}. Install the local '
                           'sam3 package and its runtime dependencies first.') from error
    if not Path(builder.__file__).resolve().is_relative_to(source):
        raise RuntimeError('Imported SAM 3 builder is not from the requested source directory.')
    # cpu build avoids the official helper treating only literal "cuda" as CUDA;
    # explicit .to(device) supports cuda:N without silently leaving the model on CPU.
    loader_output = io.StringIO()
    with contextlib.redirect_stdout(loader_output):
        model = builder.build_sam3_image_model(
            checkpoint_path=str(Path(checkpoint).resolve()), load_from_HF=False,
            device='cpu', eval_mode=True, enable_inst_interactivity=True, compile=False)
    loading_message = loader_output.getvalue()
    # The upstream loader uses strict=False and prints missing keys. Reject that
    # warning here rather than treating random/unloaded submodules as pretrained.
    if 'missing_keys=' in loading_message:
        raise RuntimeError('SAM 3 checkpoint did not populate all requested model keys: '
                           + loading_message)
    if loading_message.strip():
        print(loading_message, end='')
    model = model.to(device).eval().requires_grad_(False)
    return model, processor_module.Sam3Processor(model, device=device)


def _center(component, spacing=(1., 1.)):
    distances = distance_transform_edt(np.pad(component, 1), sampling=spacing)[1:-1, 1:-1]
    return tuple(int(v) for v in np.unravel_index(np.argmax(distances), component.shape))


def _slices(mask, maximum):
    areas = mask.sum(axis=(0, 1))
    nonempty = np.flatnonzero(areas)
    if len(nonempty) == 0:
        return []
    peak = int(np.argmax(areas))
    chosen = [peak]
    # Boundary slices already containing the target give limited extension evidence.
    # No GT, invisible target or arbitrary background point is used as an anchor.
    if maximum > 1:
        for z in (int(nonempty[0]), int(nonempty[-1])):
            if z not in chosen and len(chosen) < maximum:
                chosen.append(z)
        for z in sorted(nonempty, key=lambda z: (-int(areas[z]), int(z))):
            if int(z) not in chosen and len(chosen) < maximum:
                chosen.append(int(z))
    return chosen


def _bbox(mask, padding):
    coordinates = np.argwhere(mask)
    lo = np.maximum(coordinates.min(0) - padding, 0)
    hi = np.minimum(coordinates.max(0) + padding + 1, mask.shape)
    return lo.astype(int), hi.astype(int)


def _cap_bounds(lo, hi, anchor, maximum):
    for axis in range(2):
        if hi[axis] - lo[axis] > maximum:
            lo[axis] = max(lo[axis], min(anchor[axis] - maximum // 2, hi[axis] - maximum))
            hi[axis] = lo[axis] + maximum
    return lo, hi


def _shrink_negative(lo, hi, anchor, proposed_positive):
    """Avoid contradicting the *same SAM proposal*, not a guarantee of GT purity."""
    while proposed_positive[lo[0]:hi[0], lo[1]:hi[1]].any():
        bad = np.argwhere(proposed_positive[lo[0]:hi[0], lo[1]:hi[1]])[0] + lo
        options = []
        for axis in range(2):
            if bad[axis] < anchor[axis]:
                lower = lo.copy(); lower[axis] = bad[axis] + 1
                options.append((lower, hi.copy()))
            elif bad[axis] > anchor[axis]:
                upper = hi.copy(); upper[axis] = bad[axis]
                options.append((lo.copy(), upper))
        if not options:
            raise ValueError('Negative candidate anchor contradicts proposed background')
        lo, hi = min(options, key=lambda bounds: (-int(np.prod(bounds[1] - bounds[0])),
                                                 tuple(bounds[0]), tuple(bounds[1])))
    return lo, hi


class Sam3PromptCandidates:
    """Fixed six action slots [WT+,WT-,TC+,TC-,ET+,ET-], 2620D context.

    MRI is [4,X,Y,Z], labels [X,Y,Z] with BG/ED/NCR/ET=0/1/2/3.
    SAM receives PIL RGB slices with native X as row, Y as column. Therefore
    SAM pixel (x,y) is native (Y,X), explicitly reversed at the boundary.
    `targets` is a structured head filter, not a text prompt to SAM 3.
    """
    context_dim = CONTEXT_DIM
    slot_names = SLOT_NAMES

    def __init__(self, source_dir=None, checkpoint=None, *, device='cuda', kind='point',
                 min_component_voxels=3, slices_per_head=1, box_padding=2,
                 box_max_extent=64, proposal_margin=16, repeat_radius_mm=5.,
                 targets=HEADS, candidate_modalities_by_head=None, candidate_priority='sam_score',
                 amp=False, model=None, processor=None):
        if kind not in ('point', 'box', 'point_box'):
            raise ValueError('kind must be point, box or point_box')
        if min_component_voxels < 1 or slices_per_head < 1 or box_max_extent < 1:
            raise ValueError('Component, slice and box limits must be positive')
        if box_padding < 0 or proposal_margin < 0 or repeat_radius_mm < 0:
            raise ValueError('Padding/margin/repetition radius must be nonnegative')
        if isinstance(targets, str):
            targets = tuple(targets.split(','))
        targets = tuple(targets)
        if not targets or any(t not in HEADS for t in targets):
            raise ValueError('targets must be explicit nested heads WT/TC/ET; '
                             'ED and NCR require choosing the intended nested boundary')
        if (model is None) != (processor is None):
            raise ValueError('Provide both model and processor for injected tests')
        if candidate_priority not in ('sam_score', 'component_size') or type(amp) is not bool:
            raise ValueError('Expected sam_score or component_size priority and a boolean amp')
        if candidate_modalities_by_head is not None:
            if (not isinstance(candidate_modalities_by_head, dict)
                    or set(candidate_modalities_by_head) != set(HEADS)
                    or any(not isinstance(values, (list, tuple)) or not values
                           or any(type(v) is not int or v not in range(4) for v in values)
                           for values in candidate_modalities_by_head.values())):
                raise ValueError('candidate_modalities_by_head must map WT/TC/ET to nonempty modality index lists')
            candidate_modalities_by_head = {h: list(v) for h, v in candidate_modalities_by_head.items()}
        self.candidate_modalities_by_head = candidate_modalities_by_head
        self.candidate_priority, self.amp = candidate_priority, amp
        self.source_dir, self.checkpoint = source_dir, checkpoint
        self.device, self.kind, self.targets = str(device), kind, targets
        self.min_component_voxels, self.slices_per_head = int(min_component_voxels), int(slices_per_head)
        self.box_padding, self.box_max_extent = int(box_padding), int(box_max_extent)
        self.proposal_margin, self.repeat_radius_mm = int(proposal_margin), float(repeat_radius_mm)
        if model is None:
            model, processor = _load_official(source_dir, checkpoint, self.device)
        self.model, self.processor = model, processor
        if isinstance(model, torch.nn.Module):
            self.model.eval().requires_grad_(False)

    def config(self):
        return {'kind': self.kind, 'candidate_modalities_by_head': self.candidate_modalities_by_head,
                'candidate_priority': self.candidate_priority, 'amp': self.amp,
                'min_component_voxels': self.min_component_voxels,
                'slices_per_head': self.slices_per_head, 'box_padding': self.box_padding,
                'box_max_extent': self.box_max_extent, 'proposal_margin': self.proposal_margin,
                'repeat_radius_mm': self.repeat_radius_mm, 'targets': list(self.targets),
                'context_dim': CONTEXT_DIM, 'slot_names': list(SLOT_NAMES),
                'slot_context': 'L2-normalized 256D SAM disagreement-region pooled feature + '
                                '[present,xnorm,ynorm,znorm,SAMscore,component_plane_fraction,'
                                'component_bbox_fraction,modality_index/3]',
                'protocol': 'frozen_sam3_mask_anchored_candidates_v1'}

    def _mask_input_size(self):
        predictor = getattr(self.model, 'inst_interactive_predictor', None)
        encoder = getattr(getattr(predictor, 'model', None), 'sam_prompt_encoder', None)
        size = getattr(encoder, 'mask_input_size', None)
        if size is None:
            raise RuntimeError('SAM 3 instance predictor prompt encoder mask_input_size missing')
        return tuple(int(v) for v in size)

    def _predict(self, state, current2d):
        row, column = _center(current2d)
        lo, hi = _bbox(current2d, self.proposal_margin)
        logits = torch.from_numpy(current2d.astype(np.float32))[None, None] * 4. - 2.
        lowres = F.interpolate(logits, size=self._mask_input_size(), mode='bilinear',
                               align_corners=False)[0].numpy()
        masks, scores, _ = self.model.predict_inst(
            state, point_coords=np.array([[column, row]], dtype=np.float32),
            point_labels=np.array([1], dtype=np.int32),
            box=np.array([lo[1], lo[0], hi[1] - 1, hi[0] - 1], dtype=np.float32),
            mask_input=lowres, multimask_output=True, return_logits=False,
            normalize_coords=True)
        masks, scores = np.asarray(masks), np.asarray(scores).reshape(-1)
        if masks.ndim == 2:
            masks = masks[None]
        if masks.shape[1:] != current2d.shape or len(masks) != len(scores):
            raise ValueError('Official SAM 3 returned unexpected native-size masks/scores')
        if not np.isfinite(masks).all():
            raise ValueError('Official SAM 3 returned nonfinite mask values')
        finite = np.isfinite(scores)
        if not finite.any():
            return None, None, (lo, hi)
        chosen = int(np.argmax(np.where(finite, scores, -np.inf)))
        proposed = masks[chosen].astype(bool)
        return proposed, float(scores[chosen]), (lo, hi)

    def _candidate(self, difference, proposed, *, z, head, positive, modality, score,
                   spacing, affine, history):
        work = difference.copy()
        for previous in history:
            if previous.get('head') != head or previous.get('positive') != positive:
                continue
            coordinate = previous.get('voxel', previous.get('anchor'))
            if coordinate is None:
                continue
            rows, columns = np.where(work)
            if len(rows):
                delta = np.stack((rows - coordinate[0], columns - coordinate[1],
                                  np.full(len(rows), z - coordinate[2])), axis=1)
                world_delta = delta @ affine[:3, :3].T
                reject = np.linalg.norm(world_delta, axis=1) <= self.repeat_radius_mm
                work[rows[reject], columns[reject]] = False
        components, count = label(work)
        if count == 0:
            return None
        sizes = np.bincount(components.ravel()); sizes[0] = 0
        index = int(np.argmax(sizes))
        if sizes[index] < self.min_component_voxels:
            return None
        component = components == index
        row, column = _center(component, spacing[:2])
        anchor = [row, column, int(z)]
        prompt = {'kind': self.kind, 'head': head, 'positive': bool(positive),
                  'operation': 'include' if positive else 'exclude', 'anchor': anchor,
                  'source': 'frozen_sam3_disagreement_candidate', 'gt_geometry_used': False,
                  'sam_modality': MODALITIES[modality], 'sam_score_uncalibrated': score,
                  'component_voxels_2d': int(sizes[index])}
        if self.kind in ('point', 'point_box'):
            prompt['voxel'] = anchor.copy()
        if self.kind in ('box', 'point_box'):
            lo, hi = _bbox(component, self.box_padding)
            lo, hi = _cap_bounds(lo, hi, (row, column), self.box_max_extent)
            if not positive:
                lo, hi = _shrink_negative(lo, hi, (row, column), proposed)
            prompt.update(box_start=[int(lo[0]), int(lo[1]), int(z)],
                          box_stop=[int(hi[0]), int(hi[1]), int(z) + 1], box_axis=2,
                          box_semantics='proposal_localization_not_dense_ground_truth')
        return prompt, component

    @torch.inference_mode()
    def generate(self, mri, current, availability=None, history=(), *, affine=None):
        # Custom SAM Triton kernels launch on the current device. Tensor
        # placement alone does not select that device in a two-GPU process.
        device = torch.cuda.device(self.device) if self.device.startswith('cuda') else contextlib.nullcontext()
        with device:
            return self._generate(mri, current, availability, history, affine)

    def _generate(self, mri, current, availability, history, affine):
        from PIL import Image
        images, labels = np.asarray(mri), np.asarray(current)
        if (images.ndim != 4 or images.shape[0] != 4 or labels.ndim != 3
                or images.shape[1:] != labels.shape or not np.isfinite(images).all()
                or not np.issubdtype(labels.dtype, np.integer)
                or not np.isin(labels, (0, 1, 2, 3)).all()):
            raise ValueError('Expected finite MRI[4,X,Y,Z] and integer labels[X,Y,Z] in 0..3')
        available = np.ones(4, dtype=bool) if availability is None else np.asarray(availability, dtype=bool)
        if available.shape != (4,) or not available.any():
            raise ValueError('availability must contain four flags and at least one present modality')
        affine = np.eye(4) if affine is None else np.asarray(affine, dtype=float)
        if affine.shape != (4, 4) or not np.isfinite(affine).all() or abs(np.linalg.det(affine[:3, :3])) < 1e-12:
            raise ValueError('Expected finite invertible 4x4 voxel-to-world affine')
        spacing = np.linalg.norm(affine[:3, :3], axis=0)
        heads = [labels > index for index in range(3)]
        selected_slices = {head: _slices(heads[i], self.slices_per_head) if head in self.targets else []
                           for i, head in enumerate(HEADS)}
        all_slices = sorted({z for values in selected_slices.values() for z in values})
        actions, ranking = [None] * 6, [None] * 6
        slot_context = [np.zeros(SLOT_CONTEXT_DIM, dtype=np.float32) for _ in range(6)]
        embeddings = [[] for _ in range(4)]
        quality, disagreement = [[] for _ in HEADS], [[] for _ in HEADS]
        calls, empty_disagreement_calls = 0, 0
        for modality in range(4):
            if not available[modality]:
                continue
            nonzero = images[modality][images[modality] != 0]
            if len(nonzero):
                lower, upper = np.percentile(nonzero, (1, 99))
            else:
                lower, upper = 0., 1.
            span = max(float(upper - lower), 1e-6)
            for z in all_slices:
                gray = (np.clip((images[modality, :, :, z] - lower) / span, 0, 1) * 255).astype(np.uint8)
                rgb = np.repeat(gray[..., None], 3, axis=2)
                with torch.autocast('cuda', dtype=torch.float16,
                                    enabled=self.amp and self.device.startswith('cuda')):
                    state = self.processor.set_image(Image.fromarray(rgb))
                features = state['backbone_out']['vision_features']
                if tuple(features.shape[:2]) != (1, 256) or features.ndim != 4:
                    raise ValueError(f'Expected official SAM 3 features[1,256,H,W], got {features.shape}')
                pooled = features.float().mean(dim=(-2, -1))[0]
                pooled = F.normalize(pooled, dim=0).cpu().numpy()
                embeddings[modality].append(pooled)
                for head_index, head in enumerate(HEADS):
                    if (z not in selected_slices[head]
                            or (self.candidate_modalities_by_head is not None
                                and modality not in self.candidate_modalities_by_head[head])):
                        continue
                    current2d = heads[head_index][:, :, z]
                    with torch.autocast('cuda', dtype=torch.float16,
                                        enabled=self.amp and self.device.startswith('cuda')):
                        proposed, score, bounds = self._predict(state, current2d)
                    calls += 1
                    if proposed is None:
                        continue
                    lo, hi = bounds
                    allowed = np.zeros_like(current2d)
                    allowed[lo[0]:hi[0], lo[1]:hi[1]] = True
                    quality[head_index].append(score)
                    disagreement[head_index].append(float(((proposed != current2d) & allowed).mean()))
                    if not ((proposed != current2d) & allowed).any():
                        empty_disagreement_calls += 1
                    for polarity_index, positive in enumerate((True, False)):
                        difference = ((proposed & ~current2d) if positive else (current2d & ~proposed)) & allowed
                        proposal = self._candidate(difference, proposed, z=z, head=head,
                                                 positive=positive, modality=modality, score=score,
                                                 spacing=spacing, affine=affine, history=history)
                        if proposal is None:
                            continue
                        prompt, component = proposal
                        slot = head_index * 2 + polarity_index
                        # SAM's IoU estimate is uncalibrated on MRI; it ranks candidate
                        # evidence only. The learned utility selector decides whether to act.
                        size = prompt['component_voxels_2d']
                        evidence = (size, score) if self.candidate_priority == 'component_size' else (score, size)
                        rank = (*evidence, -modality, -z)
                        if ranking[slot] is None or rank > ranking[slot]:
                            actions[slot], ranking[slot] = prompt, rank
                            # PromptMoE's mask-aligned foundation feature idea, applied
                            # here to the actual candidate disagreement instead of the
                            # whole foreground. Area interpolation preserves tiny regions.
                            weight = torch.from_numpy(component.astype(np.float32))[None, None].to(features.device)
                            weight = F.interpolate(weight, size=features.shape[-2:], mode='area')
                            token = (features.float() * weight).sum(dim=(-2, -1))[0] / weight.sum().clamp_min(1e-12)
                            token = F.normalize(token, dim=0).cpu().numpy()
                            xyz = np.array(prompt['anchor'], dtype=float) / np.maximum(np.array(labels.shape) - 1, 1)
                            local_lo, local_hi = _bbox(component, 0)
                            geometry = [1., *xyz.tolist(), score, float(component.mean()),
                                        float(component.sum() / np.prod(local_hi - local_lo)), modality / 3.]
                            slot_context[slot] = np.concatenate((token, np.array(geometry, dtype=np.float32)))
                del state
        pooled_four = [np.mean(values, axis=0) if values else np.zeros(256, dtype=np.float32)
                       for values in embeddings]
        statistics = []
        for index, mask in enumerate(heads):
            statistics.extend([float(mask.mean()), float(mask.sum(axis=(0, 1)).max() / np.prod(mask.shape[:2])),
                               float(np.mean(quality[index])) if quality[index] else 0.,
                               max(disagreement[index], default=0.)])
        context = np.concatenate([*pooled_four, np.array(statistics, dtype=np.float32),
                                  *slot_context]).astype(np.float32)
        if context.shape != (CONTEXT_DIM,) or not np.isfinite(context).all():
            raise ValueError(f'Invalid SAM 3 context: expected finite {CONTEXT_DIM}D vector')
        return {'context': context, 'actions': actions,
                'metadata': {'slot_names': list(SLOT_NAMES), 'config': self.config(),
                             'availability': available.tolist(), 'selected_slices': selected_slices,
                             'sam_predict_calls': calls, 'gt_used': False,
                             'empty_disagreement_calls': empty_disagreement_calls,
                             'valid_slots': [action is not None for action in actions],
                             'valid_action_count': sum(action is not None for action in actions),
                             'sam_parameters_frozen': True, 'text_prompt_used': False,
                             'missing_region_limitation': [HEADS[i] for i in range(3) if not heads[i].any()],
                             'disagreement_is_not_error_ground_truth': True}}

    __call__ = generate
