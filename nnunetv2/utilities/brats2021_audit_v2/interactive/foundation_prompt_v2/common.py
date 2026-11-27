"""Data boundaries shared by collection, GT-free deployment, and scoring."""
from pathlib import Path
import numpy as np
import nibabel as nib

from ..evaluation_manifest import (read_json, write_json, check_file, file_identity,
                                    object_digest, MODALITIES, model_identity)
from ..evaluation import read_labels, same_grid
from ..interaction_metrics import region_metrics, REGIONS


def manifest(path, *, deployment=False, allowed=None):
    value = read_json(path)
    if value.get('schema') not in ('autoprompt_cases_v1', 'foundation_cases_v2'):
        raise ValueError('Expected a prepared, hash-bound case manifest')
    check_file(value['splits_file'])
    fold = read_json(value['splits_file']['path'])[value['fold']]
    train, val = set(fold['train']), set(fold['val'])
    ids = [r['case_id'] for r in value['cases']]
    if train & val or len(ids) != len(set(ids)) or not ids:
        raise ValueError('Empty, overlapping, or duplicate patient split')
    split = value['split']
    if split == 'test':
        check_file(value['test_whitelist'])
        expected = {r['case_id'] for r in read_json(value['test_whitelist']['path'])['cases']}
        if expected & (train | val):
            raise ValueError('Test overlaps development patients')
    elif split in ('train', 'validation'):
        expected = train if split == 'train' else val
    else:
        raise ValueError('Unknown split')
    if set(ids) != expected:
        raise ValueError('Patient IDs do not match the original complete split')
    if any(Path(x).name != x or x in ('.', '..') for x in ids):
        raise ValueError('Invalid patient ID')
    if allowed is not None and split not in allowed:
        raise ValueError(f'{split} is not allowed here; expected {allowed}')
    if deployment and (value.get('contains_gt') or any('gt' in r for r in value['cases'])):
        raise ValueError('Prediction requires the deployment manifest WITHOUT GT paths')
    return value


def import_manifest(source, output):
    value = manifest(source)
    destination = Path(output)
    clean = destination.with_name(destination.stem + '.deployment.json')
    if destination.exists() or clean.exists():
        raise FileExistsError('Use a new manifest path')
    value = {**value, 'schema': 'foundation_cases_v2', 'imported_from': file_identity(source)}
    write_json(destination, value)
    write_json(clean, {**value, 'contains_gt': False,
                       'cases': [{k: v for k, v in r.items() if k != 'gt'} for r in value['cases']]})
    return destination, clean


def load_case(row, *, with_gt=False):
    # No directory scanning or labelsTr/labelsTs fallback in this function.
    for identity in [row['prediction'], *row['modalities'].values()]:
        check_file(identity)
    reference = nib.load(row['prediction']['path'])
    current = read_labels(reference)
    images = []
    for name in MODALITIES:
        image = nib.load(row['modalities'][name]['path'])
        same_grid(image, reference)
        images.append(image.get_fdata(dtype=np.float32))
    images = np.stack(images)
    if not np.isfinite(images).all():
        raise ValueError('Non-finite MRI')
    if with_gt:
        check_file(row['gt'])
        truth = nib.load(row['gt']['path'])
        same_grid(truth, reference)
        return images, current, reference, read_labels(truth)
    return images, current, reference


def reward(before, after, gt):
    """Macro Dice delta in PERCENTAGE POINTS, all five regions equally weighted.

    Training only: both-empty=1 here so hallucinating an absent class incurs a
    penalty. Reporting retains undefined both-empty Dice (None).
    """
    old, new = region_metrics(before, gt), region_metrics(after, gt)
    change = np.array([100 * ((1 if new[r]['dice'] is None else new[r]['dice']) -
                              (1 if old[r]['dice'] is None else old[r]['dice'])) for r in REGIONS])
    return float(change.mean()), change


def save_mask(path, labels, reference):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(np.asarray(labels, dtype=np.uint8), reference.affine,
                           reference.header), path)
    return file_identity(path)


def source_identity():
    folder = Path(__file__).parent
    sources = {p.name: file_identity(p)['sha256'] for p in sorted(folder.glob('*.py'))}
    # The underlying correction semantics are part of the deployed method.
    for name in ('backends.py', 'local_backend.py', 'session.py', 'evaluation.py', 'schemas.py',
                 'inference_runtime.py'):
        sources['shared/' + name] = file_identity(folder.parent / name)['sha256']
    return sources


def runtime_contract(config):
    """Bind weights, candidate semantics, correction settings, and source code."""
    sam_source = Path(config['sam3']['source_dir']) / 'sam3'
    if not (sam_source / 'model_builder.py').is_file():
        raise FileNotFoundError('Requested SAM3 source tree is missing')
    source_hashes = {str(p.relative_to(sam_source)): file_identity(p)['sha256']
                     for p in sorted(sam_source.rglob('*.py'))}
    return {'config': config, 'sam3': file_identity(config['sam3']['checkpoint']),
            'sam3_source': source_hashes,
            'nninteractive': model_identity(config['correction_protocol']['model_folder']),
            'router_source': file_identity(config['router_source']),
            'implementation': source_identity()}


def signature(contract):
    return object_digest(contract)
