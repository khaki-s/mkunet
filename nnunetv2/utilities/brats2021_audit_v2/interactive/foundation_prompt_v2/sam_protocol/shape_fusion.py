"""Learn signed boundary corrections and turn them into nnInteractive prompts."""
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from scipy import ndimage
from torch import nn
from torch.nn import functional as F

from ..common import check_file, file_identity, read_json, save_mask, write_json
from . import candidates, dense, residual, spatial
from .cohorts import rows

TRAINER = 'gpu_shape_prompt'
VARIANT = 'shape_prompt'
SCHEMA = 'brats_shape_prompt_v1'
RADIUS_MM = 10.
CHANNELS = spatial.CHANNELS + 2


class ShapeReadout(spatial.SpatialReadout):
    """Predict a displacement of the original signed distance field."""

    def __init__(self):
        super().__init__()
        self.fine = nn.Conv3d(CHANNELS, 1, 1)
        self.down[0] = nn.Conv3d(CHANNELS, 12, 3, stride=2, padding=1)
        for layer in (self.fine, self.readout[-1]):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, images):
        delta = super().forward(images)
        return images[:, -2:-1].float() * RADIUS_MM + delta


def _method():
    return {**spatial._source_files(), 'shape_fusion': file_identity(__file__)['sha256']}


def signed_distance(mask, spacing):
    """Clip physical signed distance without dropping any output voxels."""
    distance = np.full(mask.shape, -RADIUS_MM, dtype=np.float16)
    if not mask.any():
        return distance
    padding = int(np.ceil(RADIUS_MM / min(spacing))) + 1
    window = residual._window(mask, padding=padding)
    inner = np.pad(mask[window], 1)
    value = ndimage.distance_transform_edt(inner, sampling=spacing)
    value -= ndimage.distance_transform_edt(~inner, sampling=spacing)
    distance[window] = np.clip(value[1:-1, 1:-1, 1:-1], -RADIUS_MM, RADIUS_MM)
    return distance


def _cache_head(key, record, output, head, labeled, method):
    basis = record['volumes']
    labels = ('baseline', 'truth') if labeled else ('baseline',)
    binding = {'method': method, 'head': head, 'inputs': {name: basis[name] for name in labels}}
    folder = Path(output) / key / dense.HEADS[head]
    path = folder / 'distance.json'
    if path.exists():
        prior = read_json(path)
        if prior['binding'] != binding:
            raise ValueError('Signed distance cache uses another source')
        for identity in prior['arrays'].values():
            check_file(identity)
        return
    folder.mkdir(parents=True, exist_ok=True)
    spacing = np.linalg.norm(np.asarray(record['affine'])[:3, :3], axis=0)
    arrays = {}
    for label in labels:
        mask = np.load(basis[label]['path'], allow_pickle=False, mmap_mode='r') > head
        arrays[label] = spatial._write_npy(folder / f'{label}.npy', signed_distance(mask, spacing))
    write_json(path, {'binding': binding, 'arrays': arrays})
    print(f'Signed distances {key} {dense.HEADS[head]} labeled={labeled}', flush=True)


def _freeze_distances(selected, volumes, destination, *, head, labeled):
    sources = spatial._index(selected, volumes, labeled=labeled)
    method = _method()
    heads = range(len(dense.HEADS)) if head is None else (head,)
    for key, record in sources.items():
        for h in heads:
            _cache_head(key, record, destination, h, labeled, method)


def prepare_shape(args):
    labeled = args.split == 'train'
    selected = rows(args.cohort, args.split, truth=labeled)[args.shard_index::args.num_shards]
    _freeze_distances(selected, args.volumes, args.output, head=args.head, labeled=labeled)
    return args.output


def _index_distances(cases, distances, *, head, labeled):
    method = _method()
    result = {}
    for key, case in cases.items():
        state = read_json(Path(distances) / key / dense.HEADS[head] / 'distance.json')
        labels = ('baseline', 'truth') if labeled else ('baseline',)
        binding = {'method': method, 'head': head, 'inputs': {name: case['volumes'][name] for name in labels}}
        if state['binding'] != binding:
            raise ValueError('Candidate correction was prepared with a different boundary')
        for identity in state['arrays'].values():
            check_file(identity)
        result[key] = (case, state['arrays'])
    return result


def _input(record, arrays, head, device, *, labeled):
    x, baseline, support, target = spatial._input(record, head, device, labeled=labeled)
    initial_distance = np.load(arrays['baseline']['path'], allow_pickle=False, mmap_mode='c')
    initial = torch.from_numpy(initial_distance).to(device).unsqueeze(0).unsqueeze(0) / RADIUS_MM
    flags = record['binding']['row']['availability']
    contrast_channel = next(index for index in dense.PREFERRED_MODALITIES[head] if flags[index])
    image = x[:, contrast_channel:contrast_channel + 1]
    affine = np.asarray(record['affine'])
    left_right = nib.aff2axcodes(affine)
    axis = next(i for i, code in enumerate(left_right) if code in ('L', 'R'))
    mirror = image.flip(dims=(2 + axis,))
    contrast = torch.where((image != 0) & (mirror != 0), image - mirror, 0.)
    inputs = torch.cat((x, initial, contrast), dim=1).contiguous(memory_format=torch.channels_last_3d)
    truth_distance = None
    if labeled:
        truth_distance = np.load(arrays['truth']['path'], mmap_mode='r', allow_pickle=False)
    return inputs, baseline, support, target, initial_distance, truth_distance


def _loss(field, baseline, target, initial_distance, truth_distance, support, device):
    near = (np.abs(initial_distance) <= 6.) | (np.abs(truth_distance) <= 6.) | (baseline != target)
    near.flat[support] = True
    indices = np.flatnonzero(near)
    if not len(indices):
        return None
    loc = torch.as_tensor(indices, device=device)
    values = field.flatten().index_select(0, loc)
    labels = torch.as_tensor(target.ravel()[indices], dtype=torch.float32, device=device)
    baseline_labels = torch.as_tensor(baseline.ravel()[indices], dtype=torch.float32, device=device)
    distances = torch.as_tensor(truth_distance.ravel()[indices], dtype=torch.float32, device=device)
    weight = 1. + (labels != baseline_labels).float()
    bce = F.binary_cross_entropy_with_logits(values / 1.4, labels, weight=weight)
    regression = (F.smooth_l1_loss(values, distances, reduction='none') * weight).mean() / 4.
    count_labels = int(np.count_nonzero(target))
    count_fixed = np.count_nonzero(baseline & ~near)
    tp_fixed = np.count_nonzero(baseline & target & ~near)
    probabilities = (values / 1.4).sigmoid()
    dice = (2. * (tp_fixed + (probabilities * labels).sum()) + 1.)
    dice = dice / (count_fixed + count_labels + probabilities.sum() + 1.)
    return bce + .5 * regression + .5 * (1. - dice)


def _network(device):
    torch.manual_seed(61119)
    return ShapeReadout().to(device=device, memory_format=torch.channels_last_3d)


def _fit(records, patients, head, epochs, device):
    model = _network(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.002, weight_decay=.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=.00015)
    scaler = torch.amp.GradScaler('cuda', init_scale=512.)
    generator = np.random.default_rng(11953 + head)
    for epoch in range(epochs):
        model.train()
        losses = []
        for i in generator.permutation(len(patients)):
            key = patients[int(i)]
            row, arrays = records[key]
            values = _input(row, arrays, head, device, labeled=True)
            x, baseline, support, target, initial, truth = values
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.float16):
                field = model(x)
            loss = _loss(field, baseline, target, initial, truth, support, device)
            if loss is None:
                continue
            if not torch.isfinite(loss):
                raise RuntimeError(f'Nonfinite signed boundary loss for {key}')
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 4.)
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach()))
        if not losses:
            raise ValueError('No labelled boundary among the training records')
        scheduler.step()
        print(f'Boundary head={head} cases={len(patients)} epoch {epoch + 1}/{epochs} '
              f'loss={np.mean(losses):.6f}', flush=True)
    return model.eval()


@torch.inference_mode()
def _probabilities(model, record, arrays, head, device, labeled):
    x, baseline, prior_support, target, _, _ = _input(record, arrays, head, device, labeled=labeled)
    with torch.autocast('cuda', dtype=torch.float16):
        field = model(x).squeeze(0).squeeze(0)
    probability = (field / 1.4).sigmoid().cpu().numpy()
    return probability, baseline, prior_support, target


def _verify_prompts(record, head, baseline, probabilities, prior, executor):
    shape = baseline.shape
    possible = ((probabilities >= .70) & ~baseline) | ((probabilities <= .30) & baseline)
    if np.count_nonzero(possible) < 8:
        return prior, []
    row = record['binding']['row']
    modality = next(i for i in dense.PREFERRED_MODALITIES[head] if row['availability'][i])
    names = tuple(dense.MODALITIES)
    modality_id = row['modalities'][names[modality]]
    check_file(modality_id)
    image = nib.load(modality_id['path']).get_fdata(dtype=np.float32)
    brain = np.load(record['volumes']['brain']['path'], mmap_mode='r', allow_pickle=False)
    proposed = baseline.copy()
    proposed[possible & brain] = probabilities[possible & brain] > .5
    spacing = np.linalg.norm(np.asarray(record['affine'])[:3, :3], axis=0)
    prompts = [candidates.locate_component(proposed, baseline, image, spacing, positive)
               for positive in (True, False)]
    if not any(prompts):
        return prior, []
    executor.engine.set_image(np.ascontiguousarray(image[None]))
    support = np.zeros(shape, dtype=bool)
    support.flat[prior] = True
    for prompt in prompts:
        if prompt is not None:
            result = executor.finite(baseline, prompt)
            support |= (result != baseline) & brain
    return np.flatnonzero(support), [p for p in prompts if p is not None]


def fit_shape(args):
    device = torch.device(args.device)
    if device.type != 'cuda' or args.epochs < 1:
        raise ValueError('Shape fitting requires a CUDA device and at least one epoch')
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    selected = rows(args.cohort, 'train', truth=True)
    h = args.head
    native = spatial._index(selected, args.volumes, labeled=True)
    records = _index_distances(native, args.distances, head=h, labeled=True)
    keys = list(records)
    calibrate = [key for i, key in enumerate(keys) if i % 5 == 4]
    fit = [key for i, key in enumerate(keys) if i % 5 != 4]
    directory = Path(args.output)
    destination = directory / f'{dense.HEADS[h]}.shape.pth'
    binding = {'schema': SCHEMA, 'trainer': TRAINER, 'method': _method(), 'head': h, 'epochs': args.epochs,
               'cohort': file_identity(args.cohort), 'n_train': len(keys), 'calibration': calibrate}
    if destination.exists():
        if torch.load(destination, map_location='cpu', weights_only=True)['binding'] != binding:
            raise ValueError('Existing shape model was trained with another protocol')
        return destination
    provisional = _fit(records, fit, h, args.epochs, device)
    torch.cuda.empty_cache()
    executor = candidates.NativeExecutor(str(device))
    gains = []
    for key in calibrate:
        row, array = records[key]
        prob, base, prior, truth = _probabilities(provisional, row, array, h, device, True)
        coords, _ = _verify_prompts(row, h, base, prob, prior, executor)
        gains.append(residual._gains(prob.ravel()[coords], coords, base, truth))
    del executor, provisional
    torch.cuda.empty_cache()
    gains = np.stack(gains)
    combined = gains.mean(axis=0) - .18 * np.maximum(0., -gains).mean(axis=0)
    options = (*residual.THRESHOLDS, None)
    i, j = np.unravel_index(np.argmax(combined), combined.shape)
    calibration = {'add': options[i], 'remove': options[j], 'calibration_gain_pp': float(gains[:, i, j].mean())}
    print(f'{dense.HEADS[h]} train-only calibration: {calibration}', flush=True)
    model = _fit(records, keys, h, args.epochs, device)
    directory.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix('.tmp')
    torch.save({'model': model.cpu().state_dict(), 'binding': binding, 'calibration': calibration}, temporary)
    temporary.replace(destination)
    write_json(destination.with_suffix('.json'), {'binding': binding, 'calibration': calibration,
              'peak_allocated_gib': torch.cuda.max_memory_allocated(device) / 2**30,
              'peak_reserved_gib': torch.cuda.max_memory_reserved(device) / 2**30})
    return destination


def _load(model_folder, cohort_path, head):
    path = Path(model_folder) / f'{dense.HEADS[head]}.shape.pth'
    model = torch.load(path, map_location='cpu', weights_only=True)
    binding = model['binding']
    if (binding['schema'] != SCHEMA or binding['method'] != _method() or binding['head'] != head
            or binding['cohort'] != file_identity(cohort_path)):
        raise ValueError('Shape model binding changed')
    return model, file_identity(path)


def load_models(model_folder, cohort):
    pairs = [_load(model_folder, cohort, h) for h in range(len(dense.HEADS))]
    return [model for model, _ in pairs], [identity for _, identity in pairs]


def infer_shape(selection, model_folder, cohort, volumes, distances, output, device):
    states, identities = load_models(model_folder, cohort)
    networks = [_network(device) for state in states]
    for network, state in zip(networks, states):
        network.load_state_dict(state['model'])
        network.eval()
    native = spatial._index(selection, volumes, labeled=False)
    indexed = [_index_distances(native, distances, head=h, labeled=False) for h in range(len(states))]
    executor = candidates.NativeExecutor(str(device)) if selection else None
    for row in selection:
        key = row['case_id']
        source, _ = indexed[0][key]
        folder = Path(output) / key
        binding = {'row': row, 'models': identities, 'selected': VARIANT, 'variants': [VARIANT],
                   'sam': source['binding']['sam'], 'responses': source['binding']['responses']}
        receipt = folder / 'prediction.json'
        if receipt.exists():
            existing = read_json(receipt)
            if existing['binding'] != binding:
                raise ValueError('Existing boundary prediction uses another model')
            saved = existing['result'][VARIANT]
            check_file(saved['mask'])
            check_file(saved['verified_wt_support'])
            continue
        outputs, edits = [], []
        for h, (network, state) in enumerate(zip(networks, states)):
            source, arrays = indexed[h][key]
            prob, baseline, prior, _ = _probabilities(network, source, arrays, h, device, False)
            coords, prompts = _verify_prompts(source, h, baseline, prob, prior, executor)
            limits = state['calibration']
            added, removed = residual._changed(prob.ravel()[coords], coords, baseline, limits['add'], limits['remove'])
            prediction = np.ascontiguousarray(baseline).copy()
            prediction.flat[coords[added]] = True
            prediction.flat[coords[removed]] = False
            outputs.append(prediction)
            edits.append({'head': dense.HEADS[h], 'prompts': prompts,
                          'added': int(added.sum()), 'removed': int(removed.sum())})
            if h == 0:
                folder.mkdir(parents=True, exist_ok=True)
                wt_support = spatial._write_npy(folder / 'verified_wt_support.npy', coords.astype(np.int32))
        reference = nib.load(row['prediction']['path'])
        path = save_mask(folder / f'{VARIANT}.nii.gz', residual.merge_heads(outputs), reference)
        result = {'mask': path, 'edits': edits, 'verified_wt_support': wt_support}
        write_json(receipt, {'binding': binding, 'gt_used': False, 'result': {VARIANT: result}})
        print(f'Native boundary prompts {key}, prompts={sum(len(edit["prompts"]) for edit in edits)}', flush=True)
    return output


def predict_shape(args):
    selected = rows(args.cohort, 'validation')[args.shard_index::args.num_shards]
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    return infer_shape(selected, args.model, args.cohort, args.volumes, args.distances, args.output, device)
