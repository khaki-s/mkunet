"""100-patient testing across every original modality subset."""
import gc
import shutil
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from scipy import ndimage

from ..common import MODALITIES, check_file, file_identity, manifest, read_json, read_labels, same_grid, write_json
from . import candidates, dense, residual, scoring
from .cohorts import read_cohort, sam_cache_spec

CODES = (15, *range(1, 15))
TEST_SCHEMA = 'brats_sam_test100_all_masks_v1'
REGION_LABELS = {'WT': (1, 2, 3), 'TC': (2, 3), 'ET': (3,), 'ED': (1,), 'NCR': (2,)}


def availability(code):
    if type(code) is not int or not 1 <= code <= 15:
        raise ValueError('Expected mask code in [1, 15]')
    return [(code >> index) & 1 for index in range(4)]


def _configuration(args):
    permit = read_json(args.permit)
    if permit.get('schema') != 'sam_test100_permit_v1':
        raise ValueError('Need formal val80 test authorization')
    check_file(permit['cohort'])
    cohort = read_cohort(permit['cohort']['path'])
    if cohort['study'] != 'formal':
        raise ValueError('Test must use train320 and val80')
    check_file(permit['val80_score'])
    report = read_json(permit['val80_score']['path'])
    if (report['study'] != 'formal' or report['n_cases'] != 80 or report['models'] != permit['models']
            or report['cohort'] != permit['cohort'] or report['variant_selected_on_train'] != permit['variant']
            or report.get('sam_contract') != permit.get('sam_contract')):
        raise ValueError('Test authorization does not match complete formal val80')
    return permit


def prepare_test(args):
    permit = _configuration(args)
    return prepare_matrix(args, permit)


def prepare_matrix(args, permit):
    deploy = manifest(args.manifest, deployment=True, allowed=('test',))
    initial = Path(args.baselines).resolve()
    provenance = read_json(initial / 'manifest.json')
    original = provenance.get('contract', {})
    patients = sorted(r['case_id'] for r in deploy['cases'])
    if (provenance.get('status') != 'complete' or len(patients) != 100
            or sorted(original.get('case_ids', [])) != patients or set(original.get('mask_codes', [])) != set(CODES)
            or original['modalities'] != list(MODALITIES)
            or original['provenance']['checkpoint_sha256'] != deploy['initial_checkpoint']['sha256']):
        raise ValueError('The complete user-specified baseline matrix must match test100')
    rows = []
    for r in deploy['cases']:
        key = r['case_id']
        existing_sources = original['provenance']['inputs'][key]
        if any(r['modalities'][m]['sha256'] != existing_sources[m]['sha256'] for m in MODALITIES):
            raise ValueError(f'{key}: baseline images differ from deployment inputs')
        predictions = {}
        for code in CODES:
            q = initial / 'predictions' / f'mask_{code:02d}' / f'{key}.nii.gz'
            predictions[str(code)] = file_identity(q)
        if predictions['15'] != r['prediction']:
            raise ValueError('mask_15 baseline differs from frozen deployment reference')
        rows.append({'case_id':key, 'modalities':r['modalities'], 'predictions':predictions})
    result = {'schema': TEST_SCHEMA, 'baseline_matrix': file_identity(initial / 'manifest.json'),
              'source':file_identity(args.manifest), 'cases':rows, 'mask_codes':list(CODES),
              'formal_permit':file_identity(args.permit), 'cohort':permit['cohort']}
    path = Path(args.output)
    if path.exists() and read_json(path) != result:
        raise ValueError('Test patient matrix would change')
    if not path.exists():
        write_json(path, result)
    return path


def test_rows(text_path, code, permit):
    index = read_json(text_path)
    check_file(index['source'])
    check_file(index['baseline_matrix'])
    check_file(index['formal_permit'])
    if (index.get('schema') != TEST_SCHEMA or index['mask_codes'] != list(CODES)
            or index['formal_permit'] != file_identity(permit)):
        raise ValueError('Need the locked test manifest for all 15 masks')
    assert code in CODES
    source = manifest(index['source']['path'], deployment=True, allowed=('test',))
    patient_ids = [r['case_id'] for r in source['cases']]
    if len(patient_ids) != 100 or [r['case_id'] for r in index['cases']] != patient_ids:
        raise ValueError('Prepared matrix does not match 100 frozen patients')
    flags = availability(code)
    return [{'case_id': r['case_id'], 'modalities': {name:r['modalities'][name]
            for i,name in enumerate(MODALITIES) if flags[i]}, 'prediction':r['predictions'][str(code)],
             'availability': flags, 'mask_code':code} for r in index['cases']]


def test_predict(args):
    permit = _configuration(args)
    phase = getattr(args, 'phase', 'all')
    trainer, variant, _, model_ids, _ = scoring.model_details(args.model, permit['cohort']['path'])
    if model_ids != permit['models'] or variant != permit['variant'] or trainer != permit.get('trainer', 'gpu_readout'):
        raise ValueError('Test model is different from the one deployed on val80')
    code = args.mask_code
    selected = test_rows(args.matrix, code, args.permit)[args.shard_index::args.num_shards]
    root = Path(args.output)
    cache = root / 'work' / f'mask_{code:02d}'
    output = root / 'predictions' / f'mask_{code:02d}'
    output.mkdir(parents=True, exist_ok=True)
    position = cache / f'shard{args.shard_index}.json'
    prior = read_json(position) if position.is_file() else {}
    measurements = {}
    if phase in ('sam', 'all'):
        cohort = read_cohort(permit['cohort']['path'])
        raw_config, factory = sam_cache_spec(read_json(cohort['train_manifest']['path']), permit.get('sam_contract'))
        if getattr(args, 'vision_only', False):
            from .image_memory import vision_factory

            factory = vision_factory(factory)
        measurements['SAM3'] = dense.build_cache(selected, args, cache/'dense', raw_config, factory=factory)
        if phase == 'all' and torch.cuda.is_available():
            gc.collect()
            torch.cuda.empty_cache()
    if phase in ('interactive', 'all'):
        measurements['nnInteractive'] = candidates.build_responses(selected, args, cache/'dense', cache/'responses')
    for name, stats in measurements.items():
        if stats['new_cases'] or name not in prior:
            prior[name] = stats
    if measurements:
        write_json(position, prior)
    if phase in ('sam', 'interactive'):
        return position
    if trainer in ('gpu_spatial', 'gpu_shape_prompt'):
        from . import spatial

        gc.collect()
        torch.cuda.empty_cache()
        device = torch.device(args.device)
        torch.cuda.set_device(device)
        spatial._freeze_volumes(selected, cache/'dense', cache/'responses', cache/'native', labeled=False)
        if trainer == 'gpu_shape_prompt':
            from . import shape_fusion

            shape_fusion._freeze_distances(selected, cache/'native', cache/'distances', head=None, labeled=False)
            shape_fusion.infer_shape(selected, args.model, permit['cohort']['path'], cache/'native',
                                     cache/'distances', cache/'predictions', device)
        else:
            spatial.infer_spatial(selected, args.model, permit['cohort']['path'], cache/'native', cache/'predictions', device)
    else:
        residual.infer(selected, args.model, permit['cohort']['path'], cache/'dense', cache/'responses', cache/'predictions')
    for row in selected:
        key = row['case_id']
        receipt = cache / 'predictions' / key / 'prediction.json'
        prediction = scoring._validate_prediction(row, receipt, model_ids, permit['variant'])
        original = prediction['result'][permit['variant']]['mask']
        check_file(original)
        path = output / f'{key}.nii.gz'
        if path.exists():
            if file_identity(path)['sha256'] != original['sha256']:
                raise ValueError(f'{key}: test artifact contains different labels')
        else:
            temp = output / f'{key}.tmp.nii.gz'
            shutil.copyfile(original['path'], temp)
            temp.replace(path)
    return output


def hd95(truth, pred, spacing):
    if not truth.any() or not pred.any():
        return None, 'both_empty' if not truth.any() and not pred.any() else 'one_empty'
    window = tuple(slice(max(0, int(v.min())-1), min(truth.shape[d], int(v.max())+2))
                   for d,v in enumerate(np.nonzero(truth | pred)))
    x, y = truth[window], pred[window]
    sx, sy = x ^ ndimage.binary_erosion(x), y ^ ndimage.binary_erosion(y)
    distances = np.concatenate((ndimage.distance_transform_edt(~sx, sampling=spacing)[sy],
                                ndimage.distance_transform_edt(~sy, sampling=spacing)[sx]))
    return float(np.percentile(distances, 95)), 'finite'


def test_score(args):
    permit = _configuration(args)
    return score_matrix_mask(args, permit)


def score_matrix_mask(args, permit, *, wt_target=None):
    index = manifest(args.truth, allowed=('test',))
    truth = {r['case_id']: r for r in index['cases']}
    root = Path(args.output)
    code = args.mask_code
    expected = test_rows(args.matrix, code, args.permit)
    result, audits = [], []
    for r in expected:
        key = r['case_id']
        y = truth[key]
        for m, source in r['modalities'].items():
            if source != y['modalities'][m]:
                raise ValueError('GT manifest differs from available images')
        path = root/'work'/f'mask_{code:02d}'/'predictions'/key/'prediction.json'
        receipt = scoring._validate_prediction(r, path, permit['models'], permit['variant'])
        artifacts = receipt['result'][permit['variant']]['mask']
        final = root/'predictions'/f'mask_{code:02d}'/f'{key}.nii.gz'
        output_identity = file_identity(final)
        if output_identity['sha256'] != artifacts['sha256']:
            raise ValueError('Exported prediction differs from native inference')
        old, gt, ref, volume = scoring._image_pair(r, y)
        image = nib.load(final)
        same_grid(image, ref)
        new = read_labels(image)
        voxel_spacing = np.linalg.norm(ref.affine[:3, :3], axis=0)
        records = scoring._measure(old, new, gt, volume)
        for row in records:
            idx = REGION_LABELS[row['region']]
            distance, state = hd95(np.isin(gt, idx), np.isin(new, idx), voxel_spacing)
            result.append({'case_id':key, 'mask_code':code, 'n_modalities':code.bit_count(),
                           **row, 'hd95_mm':distance, 'hd95_status':state})
        audits.append({'prediction_receipt':file_identity(path), 'exported':output_identity})
    directory = root/'metrics'/f'mask_{code:02d}'
    summary = {'schema':'sam_test100_subset_score_v1', 'test_matrix':file_identity(args.matrix),
               'formal_permit':file_identity(args.permit), 'mask_code':code, 'modalities':[
                   m for m, flag in zip(MODALITIES, availability(code)) if flag], 'n_cases':len(expected),
               'regions':scoring.summary(result), 'model':permit['models'], 'truth':file_identity(args.truth),
               'outputs':audits, 'per_case':result}
    write_json(directory/'summary.json', summary)
    scoring._csv(directory/'per_case_metrics.csv', result)
    # Also match the flat prediction browsing layout requested for each type.
    write_json(root/'predictions'/f'mask_{code:02d}'/'summary.json', {k:v for k,v in summary.items() if k not in ('per_case','outputs')})
    wt = summary['regions']['WT']
    if code == 15 and wt['n_paired'] == 100:
        goal = f'target {wt_target:+g} pp' if wt_target is not None else (
            f'minimum +0.5 pp, preferred {scoring.TEST_PP["WT"]:+g} pp')
        print(f'mask_15 WT over 100 cases: {100 * wt["baseline_dice_mean"]:.4f}% → '
              f'{100 * wt["dice_mean"]:.4f}% ({wt["gain_pp"]:+.4f} pp, {goal})')
    return directory/'summary.json'


def test_summary(args):
    permit = _configuration(args)
    return summarize_matrix(args, permit)


def summarize_matrix(args, permit, *, wt_target=None):
    root = Path(args.output)
    manifest_id = file_identity(args.matrix)
    reports, flat, scores = [], [], []
    for code in sorted(CODES):
        path = root/'metrics'/f'mask_{code:02d}'/'summary.json'
        summary = read_json(path)
        if (summary.get('schema')!='sam_test100_subset_score_v1' or summary['n_cases']!=100
                or summary['mask_code'] != code or summary['test_matrix'] != manifest_id
                or summary['formal_permit'] != file_identity(args.permit) or summary['model'] != permit['models']
                or len(summary['outputs']) != 100):
            raise ValueError('Need 100 independently scored cases for all 15 modality sets')
        reports.append(file_identity(path))
        scores.extend(summary['per_case'])
        for region, values in summary['regions'].items():
            flat.append({'mask_code':code, 'n_modalities':code.bit_count(), 'region':region,
                         'modalities':'+'.join(summary['modalities']), **values})
    cardinality = []
    for count in range(1, 5):
        for region in REGION_LABELS:
            subset = [r for r in flat if r['n_modalities']==count and r['region']==region]
            cardinality.append({'n_modalities':count, 'region':region, 'n_masks':len(subset),
                **{field:scoring._mean([r[field] for r in subset if r[field] is not None])
                   for field in ('baseline_dice_mean','dice_mean','gain_pp')}})
    primary = {item['region']:item for item in flat if item['mask_code']==15}
    primary_gains = {}
    accepted = {}
    for head, threshold in scoring.TEST_PP.items():
        row = primary[head]
        before, after = row['baseline_dice_mean'], row['dice_mean']
        gain = 100. * (after - before) if before is not None and after is not None else None
        primary_gains[head] = gain
        complete = row['n_cases'] == 100 and (head != 'WT' or row['n_paired'] == 100)
        accepted[head] = complete and gain is not None and gain >= threshold - 1.e-10
    wt_gain = primary_gains['WT']
    minimum = {**scoring.TEST_PP, 'WT': .5}
    wt_minimum_passed = primary['WT']['n_paired'] == 100 and wt_gain is not None and wt_gain >= minimum['WT'] - 1.e-10
    minimum_passed = {**accepted, 'WT': wt_minimum_passed}
    result = {'schema':'sam_test100_all_15_scores_v1', 'n_cases':100, 'n_masks':15,
              'test_matrix':manifest_id, 'method':permit, 'by_mask':flat, 'by_cardinality':cardinality,
              'per_mask_summaries':reports, 'primary_mask':15, 'primary_results':primary,
              'primary_gain_pp':primary_gains,
              'target_pp':scoring.TEST_PP, 'target_passed_by_head':accepted,
              'minimum_pp':minimum, 'minimum_passed_by_head':minimum_passed,
              'all_minimums_passed':all(minimum_passed.values()),
              'wt_half_pp_reached':wt_minimum_passed,
              'wt_preferred_1pp_reached':accepted['WT'],
              'all_targets_passed':all(accepted.values())}
    if wt_target is not None:
        goal_reached = primary['WT']['n_paired'] == 100 and wt_gain is not None and wt_gain >= wt_target - 1.e-10
        result.update(target_pp={'WT': wt_target}, target_passed_by_head={'WT': goal_reached})
        for key in ('minimum_pp', 'minimum_passed_by_head', 'all_minimums_passed',
                    'wt_half_pp_reached', 'wt_preferred_1pp_reached', 'all_targets_passed'):
            del result[key]
    write_json(root/'summary.json', result)
    scoring._csv(root/'summary_by_mask.csv', flat)
    scoring._csv(root/'summary_by_cardinality.csv', cardinality)
    scoring._csv(root/'per_case_metrics.csv', scores)
    return root/'summary.json'
