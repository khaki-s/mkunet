"""Align frozen multi-expert probabilities with paired native SAM3 responses."""
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage

from ..common import check_file, file_identity, read_json, read_labels, same_grid, write_json
from . import appearance, candidates, dense, expert_features, residual
from .cohorts import read_cohort, rows

EXPERTS = ('baseline', 'audit', 'moe')


def probability_path(root, expert, split, case_id):
    site = Path(root) / expert / split
    if expert == 'baseline':
        site = site / 'predictions' / 'mask_15'
    return site / f'{case_id}.json'


def probability_field(path, row, reference):
    receipt = read_json(path)
    binding = receipt['binding']
    if binding['row'] != row or receipt['gt_used'] is not False or not binding.get('probabilities_exported'):
        raise ValueError('The expert probability map must use the frozen MRI channels')
    for key in ('mask', 'probabilities'):
        check_file(receipt[key])
    image = nib.load(receipt['mask']['path'])
    same_grid(reference, image)
    labels = read_labels(image)
    with np.load(receipt['probabilities']['path'], allow_pickle=False) as archive:
        values = archive['probabilities'].transpose(0, 3, 2, 1)
    if values.shape != (len(dense.HEADS), *reference.shape) or not np.isfinite(values).all():
        raise ValueError('Unexpected model probabilities or native orientation')
    recovered = np.zeros_like(labels)
    for h, probability in enumerate(values):
        recovered[probability > .5] = h + 1
    differences = int(np.count_nonzero(labels != recovered))
    if differences > 8 or values.min() < 0 or values.max() > 1:
        raise ValueError('Native probabilities fail the model mask and threshold parity check')
    return values, labels, receipt


def _response(dense_dir, response_dir, row, shape):
    _, scores, sam = dense.read_masks(dense_dir, row)
    root = Path(response_dir) / row['case_id']
    rec = read_json(root / 'responses.json')
    if (rec['sam'] != sam['arrays'] or rec['row'] != row
            or rec['gt_used'] is not False or rec['kinds'] != list(candidates.RESPONSE_NAMES)):
        raise ValueError('Native responses must come from the stored SAM3 cue')
    check_file(rec['arrays'])
    with np.load(rec['arrays']['path'], allow_pickle=False) as source:
        packed = source['packed']
        if tuple(source['shape']) != shape or packed.shape[:3] != (2, 3, 3):
            raise ValueError('Native response geometry has changed')
    return packed, scores, sam, rec


def _member(packed, indices):
    """Read sparse native indices directly from the saved packed responses."""
    return ((packed[..., indices // 8] >> (7 - indices % 8)) & 1).astype(bool)


def _signed_support(packed, initial, brain):
    changed = np.bitwise_or.reduce(packed ^ initial, axis=0)
    allowed = np.bitwise_and(changed, brain)
    return np.flatnonzero(np.unpackbits(allowed)).astype(np.int32)


def _indices_in_prior(coordinates, source, features):
    """Match rows of recorded continuous measurements to their exact native voxels."""
    if len(source) == 0:
        return np.full((len(coordinates), 9), np.nan, dtype=np.float32)
    if len(source) != len(np.unique(source)) or features.shape[0] != len(source):
        raise ValueError('Continuous source contains duplicated native locations')
    position = np.searchsorted(source, coordinates)
    inside = position < len(source)
    inside[inside] &= source[position[inside]] == coordinates[inside]
    x = np.full((len(coordinates), 9), np.nan, dtype=np.float32)
    x[inside] = features[position[inside], -9:]
    return x


def _image_context(raw, brain, initial, p, coords, row_sign, mirror_axis):
    body = brain[::3, ::3, ::3]
    high = initial & (p[0] > .85) & (p[1] > .85) & (p[2] > .85)
    low = ~initial & (p[0] < .15) & (p[1] < .15) & (p[2] < .15)
    fields = []
    for image, present in zip(raw, row_sign):
        if not present:
            fields.append(np.full((len(coords[0]), 4), np.nan, np.float32))
            continue
        sampled = image[::3, ::3, ::3]
        normal = sampled[body & low[::3, ::3, ::3] & (sampled != 0)].astype(np.float32)
        lesion = sampled[body & high[::3, ::3, ::3] & (sampled != 0)].astype(np.float32)
        if not len(normal):
            normal = sampled[body].astype(np.float32)
        if not len(lesion):
            lesion = sampled[initial[::3, ::3, ::3]].astype(np.float32)
        normal_center = np.median(normal) if len(normal) else 0.
        lesion_center = np.median(lesion) if len(lesion) else normal_center
        scale = np.ptp(np.quantile(lesion, (.1, .9))) if len(lesion) > 4 else 1.
        value = image[coords].astype(np.float32)
        opposite = np.flip(image, axis=mirror_axis)[coords].astype(np.float32)
        fields.append(np.column_stack((value, value - normal_center,
                                       (value - lesion_center) / max(.25, scale), value - opposite)))
    return np.column_stack(fields).astype(np.float32)


def _spatial(probs, original, coords, spacing):
    # Full native geometry is retained. Only the neighborhood around sparse
    # candidate coordinates is allocated when computing local summaries.
    radius = int(np.ceil(30. / min(spacing))) + 1
    low = [max(0, a.min() - radius) for a in coords]
    high = [min(n, a.max() + radius + 1) for a, n in zip(coords, original.shape)]
    window = tuple(slice(a, b) for a, b in zip(low, high))
    local = tuple(a - b for a, b in zip(coords, low))
    actual = original[window]
    if actual.any():
        field = np.pad(actual, 1)
        d = ndimage.distance_transform_edt(field, sampling=spacing)
        d -= ndimage.distance_transform_edt(~field, sampling=spacing)
        d = np.clip(d[1:-1, 1:-1, 1:-1][local], -30., 30.)
    else:
        d = np.full(len(coords[0]), -30.)
    consensus = (probs[1][window] + probs[2][window]) * .5
    fine = ndimage.uniform_filter(consensus, size=3)[local]
    coarse = ndimage.uniform_filter(consensus, size=7)[local]
    density = ndimage.uniform_filter(actual.astype(np.float32), size=7)[local]
    return d, fine, coarse, density


def _candidate_features(head, variant, origin, brain, packed_brain, p, parts, supervised,
                        prior, raw, spacing, mirror_axis, row):
    old = origin > head
    initial = np.packbits(old.ravel())
    packed, score, sam, _ = parts
    old_native = packed[variant, head]
    union = old_native
    use_supervised = head == 0 and supervised is not None
    if use_supervised:
        new, replacement = supervised
        new_native = new[0][variant, head]
        union = np.concatenate((old_native, new_native), axis=0)
    ids = _signed_support(union, initial, packed_brain)
    if not len(ids):
        return ids, np.empty((0, 0), dtype=np.float32)
    xyz = np.unravel_index(ids, old.shape)
    base = old[xyz]
    exposure = _member(old_native, ids) != base[None, :]
    previous = _indices_in_prior(ids, prior[f'indices_{variant}_{head}'], prior[f'x_{variant}_{head}'])
    current = p[:, head]
    local = current[(slice(None), *xyz)]
    all_heads = p[(slice(None), slice(None), *xyz)]
    spatial = _spatial(current, old, xyz, spacing)
    brain_volume = int(np.count_nonzero(brain))
    prototype = _image_context(raw, brain, old, current, xyz, row['availability'], mirror_axis)
    if use_supervised:
        new_exposure = _member(new_native, ids) != base[None, :]
        cue = replacement[xyz]
        cue_score = new[1][variant, 0, xyz[2]]
        supplementary = (*new_exposure, cue, cue_score)
    else:
        supplementary = np.full((5, len(ids)), np.nan, dtype=np.float32)
    x = np.column_stack((origin[xyz], np.log1p(old.sum()) * np.ones(len(ids)),
                         np.full(len(ids), old.sum() / max(1, brain_volume)),
                         local.mean(0), local.min(0), local.max(0), local.std(0),
                         all_heads.reshape(-1, len(ids)).T, *spatial,
                         score[variant, head, xyz[2]], *exposure, *supplementary,
                         previous, prototype)).astype(np.float32)
    return ids, x


def _collect_case(args, row, split, checkpoint):
    key = row['case_id']
    labeled = split == 'train'
    clean = {k: v for k, v in row.items() if k != 'gt'}
    prob_paths = [probability_path(args.probabilities, name, split, key) for name in EXPERTS]
    if args.ready_only and not all(p.is_file() for p in prob_paths):
        return False
    folder = Path(args.output) / split / key
    path = folder / 'features.json'
    inputs = {name: file_identity(p) for name, p in zip(EXPERTS, prob_paths)}
    sources = {name: file_identity(Path(args.previous) / name / key / f'{stem}.json')
               for name, stem in (('dense', 'sam'), ('responses', 'responses'), ('features', 'features'))}
    new_dir = Path(args.supervised)
    tag = 'calibration_' if labeled else ''
    use_new = (new_dir / f'{tag}responses' / key / 'responses.json').is_file()
    if use_new:
        for name, stem in (('dense', 'sam'), ('responses', 'responses'), ('proposals', 'proposal')):
            sources['extra_' + name] = file_identity(new_dir / f'{tag}{name}' / key / f'{stem}.json')
    cache = Path(args.cache) / split / key / 'volume.json'
    binding = {'row': row, 'code': file_identity(__file__), 'volume': file_identity(cache),
               'probability_receipts': inputs, 'candidate_receipts': sources}
    if path.is_file():
        previous = read_json(path)
        if previous['binding'] != binding:
            raise ValueError('Existing voxel features belong to a different frozen source')
        check_file(previous['arrays'])
        return True
    check_file(clean['prediction'])
    reference = nib.load(clean['prediction']['path'])
    raw, old, brain, truth, lr = appearance.read_case(args.cache, row, split)
    if old.shape != reference.shape or not np.array_equal(old, read_labels(reference)):
        raise ValueError('Cached original mask differs from the frozen baseline')
    values, labels, receipts = zip(*(probability_field(p, clean, reference) for p in prob_paths))
    if receipts[0]['binding']['weights'].get('fold_0/checkpoint_best.pth') != checkpoint:
        raise ValueError('Baseline probabilities came from a different checkpoint')
    disagreement = int(np.count_nonzero(labels[0] != old))
    p = np.stack(values)
    previous_dir = Path(args.previous)
    parts = _response(previous_dir / 'dense', previous_dir / 'responses', clean, old.shape)
    prior, _ = expert_features.read_features(Path(args.previous) / 'features', clean, parts[2], parts[3])
    extra = None
    if use_new:
        new = _response(new_dir / f'{tag}dense', new_dir / f'{tag}responses', clean, old.shape)
        cue = read_json(sources['extra_proposals']['path'])
        if cue['binding']['row'] != clean or cue['gt_used'] is not False or new[2]['proposal'] != cue['arrays']:
            raise ValueError('Supervised prompt cue and response do not match')
        check_file(cue['arrays'])
        with np.load(cue['arrays']['path'], allow_pickle=False) as data:
            extra = new, data['probabilities'][0]
    packed = np.packbits(brain.ravel())
    spacing = np.linalg.norm(reference.affine[:3, :3], axis=0)
    result = {}
    if labeled:
        baseline = []
        for head in range(len(dense.HEADS)):
            original, target = old > head, truth > head
            baseline.append((int((original & target).sum()), int(original.sum() + target.sum())))
        result['baseline'] = np.asarray(baseline, dtype=np.int64)
    for v in range(len(dense.VARIANTS)):
        for h in range(len(dense.HEADS)):
            ids, x = _candidate_features(h, v, old, brain, packed, p, parts, extra, prior, raw, spacing, lr, clean)
            key_v = f'{v}_{h}'
            result[f'index_{key_v}'] = ids
            result[f'x_{key_v}'] = x.astype(np.float16)
            result[f'old_{key_v}'] = old.ravel()[ids] > h
            if labeled:
                result[f'gt_{key_v}'] = truth.ravel()[ids] > h
    folder.mkdir(parents=True, exist_ok=True)
    arrays = folder / 'features.npz'
    dense._atomic_npz(arrays, **result)
    write_json(path, {'binding': binding, 'arrays': file_identity(arrays), 'labeled': labeled,
                      'rerun_baseline_disagreement': disagreement, 'native_shape': list(old.shape)})
    print(f'{key}: saved native voxel evidence, baseline parity {disagreement} and supervised support {use_new}', flush=True)
    return True


def collect(args):
    binding = read_cohort(args.cohort)
    checkpoint = read_json(binding['train_manifest']['path'])['initial_checkpoint']
    chosen = rows(args.cohort, args.split, truth=args.split == 'train')
    for row in chosen[args.shard_index::args.num_shards]:
        _collect_case(args, row, args.split, checkpoint)
