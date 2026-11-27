"""Use independent MRI model disagreement to place paired SAM3 prompts."""
import hashlib
from pathlib import Path

import nibabel as nib
import numpy as np
from PIL import Image
from scipy import ndimage
import torch

from ...inference_runtime import NATIVE_PATCH_SIZE, prediction_context
from ..common import check_file, file_identity, manifest, read_json, read_labels, same_grid, write_json
from ..sam_dense import prompt_geometry
from . import background, candidates, dense, expert_geometry, image_memory, residual, scoring
from .cohorts import calibration_rows, rows

PROFILE = 'expert_disagreement_v7'
MULTISLICE = 'expert_multislice_v8'
EXPANDED = 'expert_expanded_support_v9'
APPEARANCE = 'native_mri_proposal_v12'
EQUIVALENT_PROFILE_CODES = {
    PROFILE: '786e1f085fd67279565acedf053f362b9e7317df7270796f31dc3669428e5174',
    MULTISLICE: '5cfa84f7a6f944fd6205c0038c76755807a9639f81b05e85220c2d713922db63',
}
EQUIVALENT_ENVELOPE_CODE = '084bd538aea7879ccd49033f8f178bc2c56610f6f28176f605c2847d67a54aef'
PRE_CALIBRATION_CODE = 'ca55d9e0041c99e9693255844f87fdf56c02f37edd82c676505b8ab00f1bfa37'
MIN_AGREEMENT = .35
COMPONENTS_PER_SIGN = (2, 1, 1)
EXPERT_RADIUS_MM = 32.


def _patients(args):
    supplied = getattr(args, 'selected_rows', None)
    if supplied is not None:
        return supplied[args.shard_index::args.num_shards]
    selected = calibration_rows(args.cohort, args.split, getattr(args, 'case_list', None))
    return selected[args.shard_index::args.num_shards]


def _same_execution(saved, current):
    if saved == current:
        return True
    previous = current.copy()
    if 'native_margin' in previous:
        previous['native_margin'] = {**previous['native_margin'], 'code': PRE_CALIBRATION_CODE}
    if 'sam_seeded_interaction_code' in previous:
        previous['sam_seeded_interaction_code'] = PRE_CALIBRATION_CODE
    return saved == previous


def _expert_votes(row, reference, directories):
    count = np.zeros((3, *reference.shape), np.uint8)
    inputs = []
    for root in directories:
        receipt = read_json(Path(root) / f'{row["case_id"]}.json')
        if receipt['gt_used'] is not False or receipt['binding']['row'] != row:
            raise ValueError('Independent expert did not use the exact label-free MRI input')
        check_file(receipt['mask'])
        image = nib.load(receipt['mask']['path'])
        same_grid(image, reference)
        result = read_labels(image)
        for h in range(3):
            count[h] += result > h
        inputs.append(receipt['mask'])
    return count, inputs


def _components(base, votes, head, spacing):
    majority = votes >= 2
    structure = ndimage.generate_binary_structure(3, 2)
    for positive, field in ((True, majority & ~base), (False, base & ~majority)):
        labels, n = ndimage.label(field, structure)
        if not n:
            continue
        sizes = np.bincount(labels.ravel(), minlength=n + 1)
        windows = ndimage.find_objects(labels)
        # Correct one spatial location at a time. The signed area from each
        # independent model is retained for the later native 3D interaction.
        selected = 0
        for k in np.argsort(sizes[1:], kind='stable')[::-1] + 1:
            if sizes[k] < 24 or selected == COMPONENTS_PER_SIGN[head]:
                break
            crop = windows[k - 1]
            mask = labels[crop] == k
            depth = ndimage.distance_transform_edt(np.pad(mask, 1), sampling=spacing)[1:-1, 1:-1, 1:-1]
            unanimity = votes[crop] == (3 if positive else 0)
            point_local = np.asarray(np.unravel_index(np.argmax(depth * (1. + .4 * unanimity)), mask.shape))
            point = point_local + [s.start for s in crop]
            component = labels == k
            preexisting = base[..., point[2]]
            plane = component[..., point[2]]
            coordinates = np.nonzero(plane)
            pad = np.maximum(1, np.ceil(10. / spacing[:2]).astype(int))
            lower = np.maximum(0, np.asarray([a.min() for a in coordinates]) - pad)
            upper = np.minimum(plane.shape, np.asarray([a.max() + 1 for a in coordinates]) + pad)
            in_plane = tuple(slice(a, b) for a, b in zip(lower, upper))
            seed = np.zeros_like(plane)
            local = preexisting | plane if positive else preexisting & ~plane
            seed[in_plane] = local[in_plane]
            if not seed.any():
                continue
            points, box = prompt_geometry(seed, padding=4)
            center_xy = point[[1, 0]].astype(np.float32)[None]
            points = np.concatenate((points, center_xy), axis=0)
            labels_sam = np.concatenate((np.ones(len(points) - 1, np.int32), np.asarray([positive], np.int32)))
            box = np.concatenate((np.minimum(box[:2], center_xy[0]), np.maximum(box[2:], center_xy[0])))
            nn_box_lo = np.maximum(lower, point[:2] - 24)
            nn_box_hi = np.minimum(upper, point[:2] + 25)
            prompt = {'point': point.tolist(), 'positive': positive,
                      'box': [[int(a), int(b)] for a, b in zip(nn_box_lo, nn_box_hi)] +
                             [[int(point[2]), int(point[2] + 1)]]}
            selected += 1
            yield component, seed, points, labels_sam, box, prompt


def _predict_pair(model, state, seed, points, labels, box, spacing, mask_size, plane, positive,
                  *, keep_masks=False):
    priors = dense.soft_prompts(seed, mask_size, spacing[:2])
    output = []
    xy = np.rint(points).astype(int)
    for prior in priors:
        mask, score, _ = model.predict_inst(state, point_coords=points, point_labels=labels, box=box,
                                            mask_input=prior, multimask_output=True, return_logits=False,
                                            normalize_coords=True)
        proposals = np.asarray(mask) > 0
        quality = np.asarray(score).reshape(-1)
        if proposals.shape != (len(quality), *seed.shape) or not np.isfinite(quality).any():
            raise ValueError('Unexpected paired SAM3 mask results')
        obey = (proposals[:, xy[:, 1], xy[:, 0]] == labels[None]).mean(1)
        best = int(np.argmax(np.where(np.isfinite(quality), quality + .25 * obey, -np.inf)))
        agree = float(np.count_nonzero((proposals[best] == positive) & plane) / np.count_nonzero(plane))
        record = {'agreement': agree, 'quality': float(quality[best]), 'accept': agree >= MIN_AGREEMENT}
        if keep_masks:
            record['mask'] = proposals[best]
        output.append(record)
    return output


def _slice_pair(model, processor, image, z, seed, points, signs, box, spacing, mask_size, plane,
                positive, *, keep_masks=False):
    gray = (image[..., z] * 255).astype(np.uint8)
    with torch.autocast('cuda', dtype=torch.float16):
        state = processor.set_image(Image.fromarray(np.repeat(gray[..., None], 3, axis=-1)))
        return _predict_pair(model, state, seed, points, signs, box, spacing, mask_size, plane,
                             positive, keep_masks=keep_masks)


def _actions(model, processor, image, base, votes, head, spacing, mask_size, multislice):
    if not multislice:
        for comp, seed, points, signs, box, prompt in _components(base, votes, head, spacing):
            z = prompt['point'][2]
            variants = _slice_pair(model, processor, image, z, seed, points, signs, box, spacing,
                                   mask_size, comp[..., z], prompt['positive'])
            yield comp, {'prompt': prompt, 'variants': variants}, [(z, points, signs, box)]
        return

    for comp, planes, positive, minority in expert_geometry.components(base, votes, head, spacing):
        evidence = [[], []]
        geometry = []
        for z, seed, points, signs, box in planes:
            component = comp[..., z]
            geometry.append((z, points, signs, box))
            variants = _slice_pair(model, processor, image, z, seed, points, signs, box, spacing,
                                   mask_size, component, positive, keep_masks=True)
            for records, result in zip(evidence, variants):
                prompt, depth = expert_geometry.native_prompt(result.pop('mask'), component, z,
                                                               positive, spacing[:2])
                records.append((prompt, depth, result))
        selected = []
        for records in evidence:
            prompt, _, best = max(records, key=lambda x: x[1] * (.5 + .5 * x[2]['agreement']))
            agreement = float(np.mean([record['agreement'] for _, _, record in records]))
            cutoff = .50 if minority else MIN_AGREEMENT
            accepted = prompt is not None and agreement >= cutoff and best['accept']
            selected.append({'agreement': agreement, 'quality': best['quality'], 'accept': accepted,
                             'prompt': prompt if accepted else None,
                             'plane_agreements': [record['agreement'] for _, _, record in records]})
        yield comp, {'positive': positive, 'minority': minority, 'variants': selected}, geometry


@torch.inference_mode()
def expert_sam(args):
    config = read_json(args.experts)
    directories = config[args.split]
    if len(directories) != 3 or len(set(directories)) != 3:
        raise ValueError('Require three independent experts')
    cohort = read_json(args.cohort)
    profile = args.profile
    multislice = profile == MULTISLICE
    contract = {**background.contract(read_json(cohort['train_manifest']['path'])), 'prompt_profile': profile,
                'expert_config': file_identity(args.experts), 'profile_code': file_identity(__file__)['sha256']}
    if multislice:
        contract['native_geometry_code'] = file_identity(expert_geometry.__file__)['sha256']
    device = torch.device(args.device)
    model = None
    for row in _patients(args):
        folder = Path(args.output) / row['case_id']
        path = folder / 'sam.json'
        if path.is_file():
            _, _, old = dense.read_masks(args.output, row)
            equivalent = {**contract, 'profile_code': EQUIVALENT_PROFILE_CODES[profile]}
            before_calibration = {**contract, 'profile_code': PRE_CALIBRATION_CODE}
            if old['contract'] not in (contract, equivalent, before_calibration):
                raise ValueError('Expert cue contract changed')
            continue
        if args.ready_only and any(not (Path(directory) / f'{row["case_id"]}.json').is_file()
                                   for directory in directories):
            continue
        mri, labels, reference = dense.load_native(row)
        votes, experts = _expert_votes(row, reference, directories)
        if model is None:
            model, processor = image_memory.load_image_only(device, stream_blocks=args.stream_blocks)
            mask_size = model.inst_interactive_predictor.model.sam_prompt_encoder.mask_input_size
        spacing = np.linalg.norm(reference.affine[:3, :3], axis=0)
        original = np.stack([labels > h for h in range(3)])
        masks = np.stack((original.copy(), original.copy()))
        score = np.zeros((2, 3, labels.shape[-1]), np.float32)
        indices = np.zeros(original.shape, np.uint8)
        digest = hashlib.sha256()
        actions = []
        sources = [next(i for i in p if row['availability'][i]) for p in dense.PREFERRED_MODALITIES]
        scaled = background._normalize(mri, set(sources))
        for h, channel in enumerate(sources):
            for comp, action, geometry in _actions(model, processor, scaled[channel], original[h], votes[h],
                                                   h, spacing, mask_size, multislice):
                ident = len(actions) + 1
                indices[h, comp] = ident
                positive = action['positive'] if multislice else action['prompt']['positive']
                for v, outcome in enumerate(action['variants']):
                    if outcome['accept']:
                        masks[v, h, comp] = positive
                    for z, *_ in geometry:
                        score[v, h, z] = max(score[v, h, z], outcome['quality'])
                for z, point, point_sign, box in geometry:
                    for array in (np.asarray([h, z, ident], np.int32), point, point_sign, box):
                        digest.update(array.tobytes())
                actions.append({'head': h, 'id': ident, **action})
        folder.mkdir(parents=True, exist_ok=True)
        arrays = folder / 'sam.npz'
        dense._atomic_npz(arrays, masks=masks, scores=score, components=indices, majority=votes >= 2)
        write_json(path, {'row': row, 'contract': contract, 'variants': list(dense.VARIANTS),
                          'arrays': file_identity(arrays), 'gt_used': False, 'expert_inputs': experts,
                          'native_shape': list(labels.shape), 'shared_point_box_digest': digest.hexdigest(),
                          'encoder_resolution': processor.resolution, 'actions': actions})
        print(f'{row["case_id"]}: {len(actions)} shared SAM3 expert-disagreement actions', flush=True)
    return args.output


def _seeded_finite(backend, base, spot, prompt):
    seed = base.astype(np.uint8)
    seed[spot] = prompt['positive']
    output = np.empty_like(seed)
    engine = backend.engine
    engine.set_target_buffer(output)
    with prediction_context(engine):
        engine.add_initial_seg_interaction(seed, run_prediction=False)
        engine.add_bbox_interaction(prompt['box'], include_interaction=prompt['positive'], run_prediction=False)
        engine.add_point_interaction(prompt['point'], include_interaction=prompt['positive'], run_prediction=True)
    refined = output.astype(bool)
    return refined | base if prompt['positive'] else refined & base


def native_margins(backend, base, indices, prompt, *, negative_point=None):
    """Read foreground minus background logits on the original patch."""
    engine = backend.engine
    destination = base.astype(np.uint8)
    engine.set_target_buffer(destination)
    with prediction_context(engine):
        engine.add_initial_seg_interaction(destination.copy(), run_prediction=False)
        if negative_point is not None:
            engine.add_point_interaction(negative_point, include_interaction=False, run_prediction=False)
        engine.add_bbox_interaction(prompt['box'], include_interaction=prompt['positive'], run_prediction=False)
        engine.add_point_interaction(prompt['point'], include_interaction=prompt['positive'], run_prediction=False)
        x, patch, window, previous = engine._build_network_input(prompt['point'], 1.)
        del previous
        engine.new_interaction_centers.clear()
        engine.new_interaction_zoom_out_factors.clear()
        if tuple(patch) != NATIVE_PATCH_SIZE or tuple(x.shape[1:]) != NATIVE_PATCH_SIZE:
            raise ValueError('Native candidate margins must use the original 192³ checkpoint patch')
        start, stop = np.asarray(window).T
        keep = np.all((indices >= start) & (indices < stop), axis=1)
        local = torch.as_tensor(indices[keep] - start, device=backend.device)
        local_index = tuple(local.T)
        with torch.autocast('cuda', dtype=torch.float16):
            logits = engine.network(x[None].contiguous())[0]
        margin = logits[(1, *local_index)] - logits[(0, *local_index)]
    return keep, margin.float().cpu().numpy()


def _margin_finite(backend, base, spot, prompt, tolerance):
    """Expose low-margin native predictions inside the independent support."""
    indices = np.argwhere(spot)
    keep, margin = native_margins(backend, base, indices, prompt)
    output = base.copy()
    output[tuple(indices[keep].T)] = margin > (-tolerance if prompt['positive'] else tolerance)
    return output


def _extend_disagreement(base, votes, core, actions, spacing, radius_mm):
    """Keep all expert disagreement near a SAM-localized point eligible."""
    expanded = core.copy()
    for h in range(len(base)):
        for positive in (True, False):
            accepted = [(a['id'], np.asarray([v['prompt']['point'] for v in a['variants'] if v['accept']]))
                        for a in actions if a['head'] == h and a['positive'] == positive
                        and any(v['accept'] for v in a['variants'])]
            if not accepted:
                continue
            support = (votes[h] > 0) & ~base[h] if positive else (votes[h] < 3) & base[h]
            locations = np.argwhere(support & (core[h] == 0))
            best = np.full(len(locations), radius_mm ** 2, dtype=np.float64)
            selected = np.zeros(len(locations), dtype=core.dtype)
            for identifier, centers in accepted:
                distances = np.square((locations[:, None, :] - centers[None, :, :]) * spacing).sum(axis=2).min(axis=1)
                better = distances < best
                best[better] = distances[better]
                selected[better] = identifier
            expanded[h][tuple(locations.T)] = selected
    return expanded


def expert_expand(args):
    dirs = read_json(args.experts)[args.split]
    if len(dirs) != 3 or len(set(dirs)) != 3:
        raise ValueError('Require three independent experts')
    for row in _patients(args):
        key = row['case_id']
        masks, scores, sam = dense.read_masks(args.dense, row)
        if sam['contract']['prompt_profile'] != MULTISLICE or sam['contract']['expert_config'] != file_identity(args.experts):
            raise ValueError('Expanded support requires the matched multi-slice SAM cues')
        contract = {**sam['contract'], 'prompt_profile': EXPANDED, 'parent_profile': MULTISLICE,
                    'envelope_code': file_identity(__file__)['sha256'],
                    'radius_mm': EXPERT_RADIUS_MM}
        folder = Path(args.output) / key
        path = folder / 'sam.json'
        if path.exists():
            _, _, previous = dense.read_masks(args.output, row)
            equivalent = {**contract, 'envelope_code': EQUIVALENT_ENVELOPE_CODE}
            before_calibration = {**contract, 'envelope_code': PRE_CALIBRATION_CODE}
            if (previous['contract'] not in (contract, equivalent, before_calibration)
                    or previous['expanded_from'] != sam['arrays']):
                raise ValueError(f'{key}: expanded support came from other SAM evidence')
            continue
        check_file(row['prediction'])
        reference = nib.load(row['prediction']['path'])
        before = read_labels(reference)
        base = np.stack([before > h for h in range(3)])
        votes, sources = _expert_votes(row, reference, dirs)
        if sources != sam['expert_inputs']:
            raise ValueError(f'{key}: independent expert masks changed')
        with np.load(sam['arrays']['path'], allow_pickle=False) as data:
            core = data['components']
            majority = data['majority']
        spacing = np.linalg.norm(reference.affine[:3, :3], axis=0)
        expanded = _extend_disagreement(base, votes, core, sam['actions'], spacing, EXPERT_RADIUS_MM)
        folder.mkdir(parents=True, exist_ok=True)
        arrays = folder / 'sam.npz'
        dense._atomic_npz(arrays, masks=masks, scores=scores, components=expanded,
                          core_components=core, majority=majority)
        write_json(path, {**sam, 'contract': contract, 'arrays': file_identity(arrays),
                          'expanded_from': sam['arrays']})
        print(f'{key}: retained {np.count_nonzero(expanded)} expert disagreement voxels near SAM3 cues', flush=True)
    return args.output


def expert_responses(args):
    if args.min_free_mib < candidates.MIN_FREE_MIB:
        raise ValueError('Requested memory budget is below the verified native 192³ entry floor')
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    if torch.cuda.mem_get_info(device)[0] < args.min_free_mib * 2**20:
        raise RuntimeError('Insufficient free CUDA memory for native 192³ responses')
    torch.cuda.reset_peak_memory_stats(device)
    execution = candidates.executor_configuration()
    seeded = args.seed_from_sam
    tolerance = args.native_margin
    if seeded and tolerance is not None:
        raise ValueError('Evaluate mask seeding and relaxed native margin in separate controls')
    if seeded:
        execution['sam_seeded_interaction_code'] = file_identity(__file__)['sha256']
    if tolerance is not None:
        execution['native_margin'] = {'tolerance': tolerance, 'code': file_identity(__file__)['sha256']}
    backend = None
    for row in _patients(args):
        raw, _, sam = dense.read_masks(args.dense, row)
        profile = sam['contract']['prompt_profile']
        if profile not in (PROFILE, MULTISLICE, EXPANDED, APPEARANCE):
            raise ValueError('Expected paired expert cues')
        if tolerance is not None and profile not in (EXPANDED, APPEARANCE):
            raise ValueError('Relaxed native margins require independent expanded support')
        execution['expert_action_code'] = sam['contract']['profile_code']
        if profile in (MULTISLICE, EXPANDED, APPEARANCE):
            execution['native_geometry_code'] = sam['contract']['native_geometry_code']
        elif 'native_geometry_code' in execution:
            del execution['native_geometry_code']
        if profile in (EXPANDED, APPEARANCE):
            execution['envelope_code'] = sam['contract']['envelope_code']
        elif 'envelope_code' in execution:
            del execution['envelope_code']
        folder = Path(args.output) / row['case_id']
        path = folder / 'responses.json'
        if path.exists():
            old = read_json(path)
            if old['row'] != row or not _same_execution(old['executor'], execution) or old['sam'] != sam['arrays']:
                raise ValueError('Existing expert responses do not match')
            check_file(old['arrays'])
            continue
        mri, label, _ = dense.load_native(row)
        base = np.stack([label > h for h in range(3)])
        with np.load(sam['arrays']['path'], allow_pickle=False) as values:
            fields = values['components']
            seed_fields = values['core_components'] if seeded and profile in (EXPANDED, APPEARANCE) else fields
        if backend is None:
            backend = candidates.NativeExecutor(str(device))
        result = np.broadcast_to(base[None, :, None], (2, 3, 3, *label.shape)).copy()
        channels = [next(i for i in p if row['availability'][i]) for p in dense.PREFERRED_MODALITIES]
        for h, channel in enumerate(channels):
            proposed = [record for record in sam['actions'] if record['head'] == h]
            if not any(v['accept'] for item in proposed for v in item['variants']):
                continue
            backend.engine.set_image(np.ascontiguousarray(mri[channel][None], dtype=np.float32))
            for action in proposed:
                selected = [v for v, evidence in enumerate(action['variants']) if evidence['accept']]
                if not selected:
                    continue
                spot = fields[h] == action['id']
                if profile == PROFILE:
                    prompt = action['prompt']
                    direction_slot = 1 if prompt['positive'] else 2
                    refined = (_seeded_finite(backend, base[h], spot, prompt) if seeded
                               else backend.finite(base[h], prompt))
                    for v in selected:
                        result[v, h, direction_slot, spot] = refined[spot]
                    continue
                direction_slot = 1 if action['positive'] else 2
                actions_run = {}
                for v in selected:
                    prompt = action['variants'][v]['prompt']
                    key = (tuple(prompt['point']), tuple(tuple(bounds) for bounds in prompt['box']))
                    if key not in actions_run:
                        if tolerance is not None:
                            actions_run[key] = _margin_finite(backend, base[h], spot, prompt, tolerance)
                        elif seeded:
                            actions_run[key] = _seeded_finite(backend, base[h], seed_fields[h] == action['id'], prompt)
                        else:
                            actions_run[key] = backend.finite(base[h], prompt)
                    refined = actions_run[key]
                    result[v, h, direction_slot, spot] = refined[spot]
            for v in range(2):
                support = raw[v, h] != base[h]
                if support.any():
                    if v == 1 and np.array_equal(raw[0, h], raw[1, h]):
                        result[1, h, 0] = result[0, h, 0]
                    else:
                        refined = backend.dense(raw[v, h])
                        result[v, h, 0, support] = refined[support]
        folder.mkdir(parents=True, exist_ok=True)
        arrays = folder / 'responses.npz'
        dense._atomic_npz(arrays, packed=np.packbits(result.reshape((2, 3, 3, -1)), axis=-1),
                          shape=np.asarray(label.shape))
        write_json(path, {'row': row, 'executor': execution, 'sam': sam['arrays'],
                          'kinds': list(candidates.RESPONSE_NAMES), 'arrays': file_identity(arrays),
                          'gt_used': False, 'modalities': channels,
                          'peak_reserved_gib': torch.cuda.max_memory_reserved(device) / 2**30})
        print(f'{row["case_id"]}: expert prompts completed with original 192³ nnInteractive', flush=True)
    return args.output


def expert_probe_score(args):
    selected = rows(args.cohort, 'validation')
    targets = {row['case_id']: row for row in manifest(args.truth, allowed=('validation',))['cases']}
    measurements, coverage, actions = {}, [], []
    for row in selected:
        key = row['case_id']
        before, truth, _, volume = scoring._image_pair(row, targets[key])
        base = np.stack([before > h for h in range(3)])
        masks, _, sam = dense.read_masks(args.dense, row)
        native, ref = candidates.read_responses(args.responses, row)
        if sam['arrays'] != ref['sam']:
            raise ValueError('Native responses came from different expert cues')
        with np.load(sam['arrays']['path'], allow_pickle=False) as cube:
            majority = cube['majority']
        estimates = {'expert_majority': majority}
        for i, variant in enumerate(dense.VARIANTS):
            additions, removals = native[i, :, 1], native[i, :, 2]
            point = residual.combine_native_point_box(base, additions, removals)
            estimates[f'{variant}/expert_sam_gate'] = masks[i]
            estimates[f'{variant}/native_dense'] = native[i, :, 0]
            estimates[f'{variant}/native_point_box'] = point
            estimates[f'{variant}/native_add_only'] = additions
            estimates[f'{variant}/native_remove_only'] = removals
            estimates[f'{variant}/native_agreement'] = np.where(point == native[i, :, 0], point, base)
            for head, view in enumerate(dense.HEADS):
                support = np.any(native[i, head] != base[head], axis=0)
                fields = residual._coverage(base[head], truth > head, support)
                coverage.append({'case_id': key, 'variant': variant, 'region': view, **fields})
        actions.extend({'case_id': key, **action} for action in sam['actions'])
        visible = (before > 0) | (truth > 0)
        for result in estimates.values():
            visible |= result[0]
        # Background outside this box makes no contribution to any counts.
        window = residual._window(visible, padding=0) if visible.any() else np.s_[:, :, :]
        before_crop, truth_crop = before[window], truth[window]
        for name, result in estimates.items():
            labels = residual.merge_heads(result[(slice(None), *window)])
            measurements.setdefault(name, []).extend({'case_id': key, **item}
                for item in scoring._measure(before_crop, labels, truth_crop, volume))
    report = {'scope': 'fixed validation patients, no fitted selector; diagnostic only',
              'n_cases': len(selected), 'comparison': {k: scoring.summary(v) for k, v in measurements.items()},
              'native_candidate_coverage': coverage, 'actions': actions, 'per_case': measurements}
    write_json(args.output, report)
    return args.output
