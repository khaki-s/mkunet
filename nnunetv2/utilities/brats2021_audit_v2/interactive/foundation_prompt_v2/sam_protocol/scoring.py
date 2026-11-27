"""Independent paired Dice and per-subset BraTS output summaries."""
import csv
from collections import defaultdict
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage

from ...interaction_metrics import REGIONS, region_metrics
from ..common import MODALITIES, check_file, file_identity, manifest, read_json, read_labels, same_grid, write_json
from . import dense, residual, candidates
from .cohorts import VAL_COUNT, read_cohort, rows

PILOT_WT_PP = 1.0
TEST_PP = {'WT': 1., 'TC': .5, 'ET': .5}
GPU_TRAINERS = {'gpu_readout', 'gpu_spatial', 'gpu_shape_prompt'}


def _mean(values):
    return float(np.mean(values)) if values else None


def _csv(path, records):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    with temp.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    temp.replace(path)


def _measure(before, after, truth, voxel_volume):
    old = region_metrics(before, truth, voxel_volume_mm3=voxel_volume)
    new = region_metrics(after, truth, voxel_volume_mm3=voxel_volume)
    rows_ = []
    for label in REGIONS:
        a, b = old[label], new[label]
        paired = None if a['dice'] is None or b['dice'] is None else 100 * (b['dice'] - a['dice'])
        rows_.append({'region': label, 'baseline_dice': a['dice'], 'dice': b['dice'], 'gain_pp': paired,
                      'ref_mm3': b['n_ref'] * voxel_volume, 'pred_mm3': b['n_pred'] * voxel_volume,
                      'baseline_fp_mm3': a['fp_mm3'], 'baseline_fn_mm3': a['fn_mm3'],
                      'fp_mm3': b['fp_mm3'], 'fn_mm3': b['fn_mm3']})
    return rows_


def summary(records):
    output = {}
    for name in REGIONS:
        subset = [r for r in records if r['region'] == name]
        before = [r['baseline_dice'] for r in subset if r['baseline_dice'] is not None]
        after = [r['dice'] for r in subset if r['dice'] is not None]
        paired = [r['gain_pp'] for r in subset if r['gain_pp'] is not None]
        empty = [r for r in subset if r['ref_mm3'] == 0]
        output[name] = {'n_cases': len(subset), 'n_valid_dice': len(after), 'n_paired': len(paired),
                         'baseline_dice_mean': _mean(before), 'dice_mean': _mean(after),
                         'gain_pp': _mean(paired), 'dice_median': float(np.median(after)) if after else None,
                         'dice_p10': float(np.quantile(after, .1)) if after else None,
                         'n_improved': sum(v>1.e-8 for v in paired), 'n_worsened': sum(v<-1.e-8 for v in paired),
                         'n_empty_gt': len(empty), 'n_dice_below_0_5': sum(v<.5 for v in after),
                         'nonempty_gt_dice_mean': _mean([r['dice'] for r in subset if r['ref_mm3'] > 0]),
                         'empty_gt_fp_mm3_mean': _mean([r['fp_mm3'] for r in empty]),
                         'empty_gt_fp_increase_mm3': sum(max(0, r['fp_mm3'] - r['baseline_fp_mm3']) for r in empty)}
        if subset and 'hd95_status' in subset[0]:
            distances = [r['hd95_mm'] for r in subset if r['hd95_status'] == 'finite']
            output[name].update(hd95_finite_mean_mm=_mean(distances), n_hd95_finite=len(distances),
                n_hd95_one_empty=sum(r['hd95_status']=='one_empty' for r in subset),
                n_hd95_both_empty=sum(r['hd95_status']=='both_empty' for r in subset))
    return output


def _validate_prediction(row, path, model_ids, variant):
    record = read_json(path)
    binding = record['binding']
    if (record.get('gt_used') is not False or binding['row'] != row or binding['models'] != model_ids
            or binding['selected'] != variant):
        raise ValueError('Mismatch in completed label-free prediction')
    return record


def _image_pair(row, truth_row):
    check_file(row['prediction'])
    check_file(truth_row['gt'])
    image = nib.load(row['prediction']['path'])
    label = nib.load(truth_row['gt']['path'])
    same_grid(image, label)
    volume = float(abs(np.linalg.det(image.affine[:3,:3])))
    return read_labels(image), read_labels(label), image, volume


def _image_difficulty(row, before, target, reference):
    """Label-aided review of missed slices and MRI boundary contrast."""
    gt = target > 0
    baseline = before > 0
    true_slices = gt.any(axis=(0, 1))
    missed = int(np.count_nonzero(true_slices & ~baseline.any(axis=(0, 1))))
    summary = {'entirely_missed_wt_slices': missed, 'flair_wt_boundary_cnr': None,
               't2_wt_boundary_cnr': None, 'wt_volume_ml': None}
    if not gt.any():
        return summary
    voxel_volume = float(abs(np.linalg.det(reference.affine[:3, :3])))
    summary['wt_volume_ml'] = float(gt.sum() * voxel_volume / 1000.)
    extent = residual._window(gt, padding=7)
    region = gt[extent]
    sampling = np.linalg.norm(reference.affine[:3, :3], axis=0)
    outside = ndimage.distance_transform_edt(~region, sampling=sampling)
    inside = ndimage.distance_transform_edt(np.pad(region, 1), sampling=sampling)[1:-1, 1:-1, 1:-1]
    for modality in ('FLAIR','T2'):
        identity = row['modalities'][modality]
        check_file(identity)
        image = nib.load(identity['path'])
        same_grid(image, reference)
        values = np.asarray(image.dataobj[extent], dtype=np.float32)
        inner = values[region & (inside <= 4.) & (values != 0)]
        outer = values[(outside > 0.) & (outside <= 4.) & (values != 0)]
        if len(inner) and len(outer):
            mid = np.median(outer)
            robust_scale = 1.4826 * np.median(np.abs(outer - mid))
            if robust_scale > 1.e-6:
                summary[f'{modality.lower()}_wt_boundary_cnr'] = float((np.median(inner) - mid) / robust_scale)
    return summary


def model_details(root, cohort):
    if (Path(root) / 'WT.shape.pth').is_file():
        from . import shape_fusion as native
    elif (Path(root) / 'WT.pth').is_file():
        from . import spatial as native
    else:
        native = None

    if native is not None:
        model, ids = native.load_models(root, cohort)
        epochs = {state['binding']['epochs'] for state in model}
        if len(epochs) != 1 or any(state['binding']['trainer'] != native.TRAINER for state in model):
            raise ValueError('Spatial heads must use the same GPU training settings')
        return native.TRAINER, native.VARIANT, model[0]['calibration'], ids, {'epochs': epochs.pop()}
    model, index, ids = residual.load_models(root, cohort)
    trainers = {m['binding'].get('trainer', 'cpu_histogram') for m in model}
    weighting = {m['binding'].get('case_weighting') for m in model}
    if len(trainers) != 1 or len(weighting) != 1:
        raise ValueError('Expected one training method for all three heads')
    case_weighting = weighting.pop()
    settings = {'case_weighting': case_weighting} if case_weighting is not None else {}
    if model[0]['binding'].get('feature_schema'):
        settings['feature_schema'] = model[0]['binding']['feature_schema']
    return trainers.pop(), dense.VARIANTS[index], model[0]['variants'][index], ids, settings


def score_validation(args):
    cohort = read_cohort(args.cohort)
    selected = rows(args.cohort, 'validation')
    validation = manifest(args.truth, allowed=('validation',))
    truth_rows = {r['case_id']:r for r in validation['cases']}
    trainer, variant, calibration, ids, settings = model_details(args.model, args.cohort)
    native = trainer in ('gpu_spatial', 'gpu_shape_prompt')
    views = (variant,) if native else dense.VARIANTS
    output = defaultdict(list)
    diagnostic, receipts = [], []
    raw_config = None
    for row in selected:
        key = row['case_id']
        label = truth_rows[key]
        expected = {k: val for k,val in row.items() if k != 'mask_code'}
        if {k: val for k,val in label.items() if k != 'gt'} != expected:
            raise ValueError('Validation GT and deployment rows disagree')
        path = Path(args.predictions) / key / 'prediction.json'
        record = _validate_prediction(row, path, ids, variant)
        receipts.append(file_identity(path))
        old, target, ref, volume = _image_pair(row, label)
        image_features = _image_difficulty(row, old, target, ref)
        sam, _, sam_receipt = dense.read_masks(args.dense, row)
        response, nn_receipt = candidates.read_responses(args.responses, row)
        if record['binding']['sam'] != sam_receipt['arrays'] or nn_receipt['sam'] != sam_receipt['arrays']:
            raise ValueError('Scoring requires the exact SAM artifact used by validation')
        if record['binding']['responses'] != nn_receipt['arrays']:
            raise ValueError('Scoring requires the exact nnInteractive artifact used by validation')
        if settings.get('feature_schema'):
            check_file(record['binding']['features'])
        if raw_config is None:
            raw_config = sam_receipt['contract']
        elif raw_config != sam_receipt['contract']:
            raise ValueError('Validation mixes different SAM prompt profiles')
        for i, name in enumerate(views):
            mask = record['result'][name]['mask']
            check_file(mask)
            image = nib.load(mask['path'])
            same_grid(image, ref)
            new = read_labels(image)
            output[name].extend({'case_id': key, 'mask_code':15, 'n_modalities':4, 'variant':name, **r}
                                for r in _measure(old, new, target, volume))
            baseline, gt = old > 0, target > 0
            if native:
                combined = np.any(response[:, 0] != baseline, axis=(0, 1))
                sam_changed = np.any(sam[:, 0] != baseline, axis=0)
                verified = record['result'][name].get('verified_wt_support')
                if verified is not None:
                    check_file(verified)
                    combined.flat[np.load(verified['path'], allow_pickle=False)] = True
            else:
                combined = np.any(response[i, 0] != baseline, axis=0)
                sam_changed = sam[i, 0] != baseline
            coverage = residual._coverage(baseline, gt, combined)
            coverage['sam_oracle_gain_pp'] = residual._coverage(baseline, gt, sam_changed)['oracle_gain_pp']
            coverage['gt_components'] = int(ndimage.label(gt)[1])
            coverage['baseline_components'] = int(ndimage.label(baseline)[1])
            coverage['largest_true_FN_component_voxels'] = _largest(gt & ~baseline)
            coverage['largest_true_FP_component_voxels'] = _largest(baseline & ~gt)
            diagnostic.append({'case_id':key, 'variant':name, **coverage, **image_features})
    by_variant = {key:summary(recs) for key,recs in output.items()}
    report = {'schema':'sam_paired_val_v1', 'study':cohort['study'], 'cohort':file_identity(args.cohort),
              'n_cases':len(selected), 'predictions':receipts, 'models':ids, 'variant_selected_on_train':variant,
              'regions':by_variant[variant], 'by_variant':by_variant, 'target_WT_pp':PILOT_WT_PP,
              'trainer':trainer, 'trainer_settings':settings, 'sam_contract':raw_config,
              'selected_train_calibration_pp':calibration['calibration_gain_pp'],
              'selected_case_ids':cohort['validation'], 'truth_manifest':file_identity(args.truth)}
    report['pilot_gate_passed'] = (cohort['study']=='pilot' and report['regions']['WT']['n_paired']==VAL_COUNT['pilot']
                                  and report['regions']['WT']['gain_pp'] >= PILOT_WT_PP)
    root = Path(args.output)
    write_json(root / 'score.json', report)
    _csv(root / 'per_case_metrics.csv', [item for variant_rows in output.values() for item in variant_rows])
    _csv(root / 'wt_candidate_coverage.csv', sorted(diagnostic, key=lambda row:row['baseline_pct']))
    return root / 'score.json'


def _largest(mask):
    labels, n = ndimage.label(mask)
    return int(np.bincount(labels.ravel(), minlength=n+1)[1:].max(initial=0))


def _validated_report(path, mode, *, cohort_path=None, models=None):
    report = read_json(path)
    if (report.get('schema') != 'sam_paired_val_v1' or report['study'] != mode
            or report['n_cases'] != VAL_COUNT[mode] or len(report['predictions']) != VAL_COUNT[mode]):
        raise ValueError(f'Need all {VAL_COUNT[mode]} frozen validation predictions')
    check_file(report['cohort'])
    cohort = read_cohort(report['cohort']['path'])
    if cohort['study'] != mode or report['selected_case_ids'] != cohort['validation']:
        raise ValueError('Validation report does not match the ordered frozen cohort')
    if cohort_path is not None and report['cohort'] != file_identity(cohort_path):
        raise ValueError('Different training cohort')
    if models is not None and report['models'] != models:
        raise ValueError('Different frozen models')
    seen = []
    for identity in report['predictions']:
        check_file(identity)
        record = read_json(identity['path'])
        binding = record['binding']
        if record.get('gt_used') is not False or 'gt' in binding['row'] or binding['models'] != report['models']:
            raise ValueError('Validation receipt includes GT or another checkpoint')
        seen.append(binding['row']['case_id'])
        for result in record['result'].values():
            check_file(result['mask'])
    if seen != cohort['validation']:
        raise ValueError('Incomplete or duplicated patient receipts')
    return report


def admit(args):
    report = _validated_report(args.score, 'pilot')
    wt = report['regions']['WT']
    if wt['n_paired'] != VAL_COUNT['pilot'] or wt['gain_pp'] is None or wt['gain_pp'] < PILOT_WT_PP:
        raise ValueError(f'Formal training locked: pilot WT={wt["gain_pp"]} pp; requires at least {PILOT_WT_PP:g} pp')
    if report.get('trainer') not in GPU_TRAINERS:
        raise ValueError('Formal multi-GPU training requires a passing GPU-trained pilot for this readout')
    check_file(report['truth_manifest'])
    truth = manifest(report['truth_manifest']['path'], allowed=('validation',))
    key_to_truth = {row['case_id']: row for row in truth['cases']}
    gains = []
    variant = report['variant_selected_on_train']
    for identity in report['predictions']:
        prediction = read_json(identity['path'])
        row = prediction['binding']['row']
        y = key_to_truth[row['case_id']]
        before, target, reference, volume = _image_pair(row, y)
        mask = nib.load(prediction['result'][variant]['mask']['path'])
        same_grid(mask, reference)
        gains.append(_measure(before, read_labels(mask), target, volume)[0]['gain_pp'])
    if any(v is None for v in gains) or not np.isclose(np.mean(gains), wt['gain_pp'], atol=1.e-9, rtol=0):
        raise ValueError('Saved validate25 masks do not reproduce the WT gate')
    permission = {'schema':'sam_formal_training_permit_v1', 'pilot_score':file_identity(args.score), 'trainer':report['trainer'],
                  'trainer_settings':report.get('trainer_settings', {}), 'sam_contract':report.get('sam_contract'),
                  'wt_gain_pp':wt['gain_pp'], 'variant':report['variant_selected_on_train']}
    write_json(args.output, permission)
    return args.output


def require_training_permit(path):
    if not path or not Path(path).is_file():
        raise ValueError('Formal training requires the passing, verified validate25 report')
    permit = read_json(path)
    if permit.get('schema') != 'sam_formal_training_permit_v1':
        raise ValueError('Invalid formal-start permission')
    check_file(permit['pilot_score'])
    pilot = _validated_report(permit['pilot_score']['path'], 'pilot')
    wt = pilot['regions']['WT']
    if (wt['n_paired'] != VAL_COUNT['pilot'] or wt['gain_pp'] != permit['wt_gain_pp']
            or wt['gain_pp'] < PILOT_WT_PP or permit.get('trainer') not in GPU_TRAINERS
            or pilot.get('trainer') != permit['trainer']
            or pilot.get('trainer_settings', {}) != permit.get('trainer_settings', {})
            or pilot.get('sam_contract') != permit.get('sam_contract')):
        raise ValueError('Pilot validation threshold not met')
    return permit


def permit_test(args):
    cohort = read_cohort(args.cohort)
    if cohort['study'] != 'formal':
        raise ValueError('Test requires a train320 and val80 cohort')
    authorized = require_training_permit(cohort.get('pilot_permit', {}).get('path'))
    trainer, variant, _, ids, settings = model_details(args.model, args.cohort)
    report = _validated_report(args.score, 'formal', cohort_path=args.cohort, models=ids)
    if (report['trainer'] != trainer or trainer != authorized['trainer']
            or report['variant_selected_on_train'] != variant
            or report.get('trainer_settings', {}) != settings or authorized.get('trainer_settings', {}) != settings):
        raise ValueError('Different fixed soft prompt model')
    if authorized.get('sam_contract') is not None and report.get('sam_contract') != authorized['sam_contract']:
        raise ValueError('Formal validation did not use the approved SAM profile')
    permit = {'schema':'sam_test100_permit_v1', 'cohort':file_identity(args.cohort), 'models':ids,
              'trainer':trainer, 'variant':variant, 'sam_contract':report.get('sam_contract'),
              'val80_score':file_identity(args.score), 'study':'formal'}
    write_json(args.output, permit)
    return args.output
