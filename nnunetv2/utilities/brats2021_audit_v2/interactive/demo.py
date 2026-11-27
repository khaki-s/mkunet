"""Frozen nnInteractive + Qwen pilot. Manual inference never loads ground truth.

Hard-label initialization avoids rerunning LegacyV2 just to export probabilities.
The +/- logit constants encode binary membership, NOT recovered confidence.
GT access is confined to the explicit simulate/score commands.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
from http.server import BaseHTTPRequestHandler, HTTPServer
import importlib.metadata
import importlib.util
import io
import json
from pathlib import Path
import secrets
import sys
from urllib.parse import parse_qs, urlparse

import nibabel as nib
import numpy as np
from PIL import Image
import torch

from .backends import NNInteractiveBackend
from .atomic import atomic_point_transaction
from .controller import RuleController, LLMController
from .schemas import BoxPrompt, PointPrompt, SpatialROI, REGIONS
from .session import CorrectionSession

DEFAULT_MODEL = Path('/data02/chujingyi/model/nnInteractive/nnInteractive_v1.0')
DEFAULT_QWEN = Path('/data02/chujingyi/model/Qwen2.5-7B-Instruct')
MODALITIES = ('T1', 'T1ce', 'T2', 'FLAIR')


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def digest(path):
    sha = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b''):
            sha.update(chunk)
    return sha.hexdigest()


def internal_labels(image):
    labels = np.asanyarray(image.dataobj)
    if labels.ndim != 3 or not np.isin(labels, (0, 1, 2, 3)).all():
        raise ValueError('Require internal labels 0=BG, 1=ED, 2=NCR, 3=ET; do not pass raw BraTS label 4')
    return labels.astype(np.uint8)


def labels_from_logits(logits):
    values = np.asarray(logits)
    output = np.zeros(values.shape[1:], np.uint8)
    for head, label in enumerate((1, 2, 3)):
        output[values[head] > 0] = label
    return output


def preflight(args):
    dependencies = {}
    for distribution, module in (('torch', 'torch'), ('nnInteractive', 'nnInteractive'),
                                 ('transformers', 'transformers'), ('nibabel', 'nibabel'),
                                 ('scipy', 'scipy'), ('Pillow', 'PIL')):
        try:
            version = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            version = None
        dependencies[distribution] = {'version': version, 'import_available': importlib.util.find_spec(module) is not None}
    folder = args.nninteractive_model
    files = {name: (folder / name).is_file() for name in
             ('plans.json', 'dataset.json', 'fold_0/checkpoint_final.pth')}
    qwen = args.llm_path
    qwen_index = qwen / 'model.safetensors.index.json'
    shards = []
    if qwen_index.is_file():
        shards = sorted(set(json.loads(qwen_index.read_text()).get('weight_map', {}).values()))
    llm_files = {name: (qwen / name).is_file() for name in
                 ('config.json', 'tokenizer_config.json', 'tokenizer.json', *shards)}
    required = ('torch', 'nnInteractive', 'nibabel', 'scipy', 'Pillow')
    blockers = [f'Missing package: {name}' for name in required if not dependencies[name]['import_available']]
    if not all(files.values()):
        blockers.append('Incomplete nnInteractive model folder')
    if args.require_qwen and (not dependencies['transformers']['import_available'] or
                             not shards or not all(llm_files.values())):
        blockers.append('Qwen requires transformers and all local config/tokenizer/weight shards')
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        blockers.append('CUDA unavailable to this process; check nvidia-smi/driver and GPU allocation')
    # Import the actual engine, not only the namespace package.
    if dependencies['nnInteractive']['import_available']:
        try:
            from nnInteractive.inference.inference_session import nnInteractiveInferenceSession
            dependencies['nnInteractive']['engine_import'] = True
        except Exception as error:
            blockers.append(f'nnInteractive engine import failed: {type(error).__name__}: {error}')
    report = {'python': sys.version, 'executable': sys.executable, 'dependencies': dependencies,
              'cuda_available': torch.cuda.is_available(), 'cuda_count': torch.cuda.device_count(),
              'nninteractive_model': str(folder), 'nninteractive_files': files,
              'qwen_model': str(qwen), 'qwen_files': llm_files,
              'ready': not blockers, 'blockers': blockers, 'weights_loaded': False}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        write_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report['ready'] else 2


def prepare(args):
    if args.output.exists():
        raise FileExistsError('Use a new case file; original files are never overwritten')
    image = nib.load(str(args.prediction))
    labels = internal_labels(image)
    paths = [getattr(args, name.lower()) for name in MODALITIES]
    present = np.array([p is not None for p in paths])
    if not present.any():
        raise ValueError('Supply at least one available raw MRI modality')
    raw = np.zeros((4, *labels.shape), np.float32)
    for channel, path in enumerate(paths):
        if path is None:
            continue
        mri = nib.load(str(path))
        if mri.shape != labels.shape or not np.allclose(mri.affine, image.affine, atol=1e-5, rtol=0):
            raise ValueError('MRI/prediction shape or affine mismatch; do not resize labels implicitly')
        raw[channel] = mri.get_fdata(dtype=np.float32)
    if not np.isfinite(raw).all():
        raise ValueError('MRI contains nonfinite values')
    regions = np.stack((labels > 0, labels >= 2, labels == 3))
    logits = np.where(regions, 4.59512, -4.59512).astype(np.float32)
    # The official backend consumes raw_mri. This normalization is only for the
    # common session's unused neural-corrector input, not for nnInteractive.
    normalized = np.zeros_like(raw)
    for channel in np.flatnonzero(present):
        foreground = raw[channel] != 0
        if foreground.any():
            values = raw[channel][foreground]
            normalized[channel][foreground] = (values - values.mean()) / max(float(values.std()), 1e-8)
    metadata = {'initialization': 'hard_labels_only_not_calibrated_probabilities',
                'axis_order': 'nibabel_native_xyz', 'regions': REGIONS,
                'source_prediction': str(args.prediction.resolve()),
                'source_prediction_sha256': digest(args.prediction),
                'source_mri': [str(p.resolve()) if p else None for p in paths],
                'gt_loaded': False, 'model_parameters_updated': False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('xb') as stream:
        np.savez_compressed(stream, mri=normalized, raw_mri=raw, logits=logits,
                            affine=image.affine, modality_mask=present,
                            case_id=np.array(args.case_id), split=np.array(args.split),
                            metadata=np.array(json.dumps(metadata)))
    print(json.dumps({'case': str(args.output), 'shape': labels.shape, **metadata}, ensure_ascii=False))


def load_case(path):
    with np.load(path, allow_pickle=False) as archive:
        # Even if a train preparation accidentally reaches this UI, no labels
        # are admitted to the manual inference process.
        if any(key in archive for key in ('atomic_labels', 'ground_truth', 'region_gt', 'gt', 'training_labels')):
            raise ValueError('Manual demo accepts GT-free case bundles only')
        keys = ('mri', 'raw_mri', 'logits', 'affine', 'modality_mask', 'case_id', 'split')
        result = {key: np.array(archive[key]) for key in keys}
        result['metadata'] = json.loads(str(archive['metadata'].item())) if 'metadata' in archive else {}
    return result


def make_session(args):
    case = load_case(args.case)
    if not bool(case['modality_mask'][args.modality]):
        raise ValueError('Selected nnInteractive modality is missing')
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; run preflight on the GPU host before serve/simulate')
    corrector = NNInteractiveBackend(args.nninteractive_model,
        case['raw_mri'][args.modality:args.modality + 1], case['affine'],
        modality_index=args.modality, device=args.device)
    session = CorrectionSession(case['mri'], case['modality_mask'], case['logits'],
        case['affine'], corrector, case_id=str(case['case_id'].item()), split=str(case['split'].item()),
        source_provenance=case['metadata'])
    controller = RuleController()
    if args.controller == 'qwen':
        from .llm import LocalQwenGenerator
        controller = LLMController(LocalQwenGenerator(args.llm_path, device=args.llm_device))
    return session, controller, case


def bounded_roi(points, box, shape, size):
    positions = [tuple(p['voxel']) for p in points]
    if box is not None:
        positions += [tuple(box['start']), tuple(int(v) - 1 for v in box['stop'])]
    if not positions:
        raise ValueError('Select at least one positive/negative point or a slice box')
    if any(len(p) != 3 or any(type(v) is not int or v < 0 or v >= shape[a]
                              for a, v in enumerate(p)) for p in positions):
        raise ValueError('Prompt outside the native image grid')
    lo = np.min(positions, axis=0)
    hi = np.max(positions, axis=0) + 1
    extent = np.minimum(shape, np.maximum(size, hi - lo + 16))
    start = np.clip((lo + hi - extent) // 2, 0, np.array(shape) - extent)
    return SpatialROI(tuple(int(v) for v in start), tuple(int(v) for v in start + extent))


class DemoState:
    def __init__(self, session, controller, raw, output, *, case_path, controller_name, roi_size=96):
        self.session, self.controller, self.raw = session, controller, raw
        self.output, self.case_path = Path(output), Path(case_path)
        self.controller_name, self.roi_size = controller_name, roi_size
        self.serial = 0
        self.commands = []
        self.export_serial = 0

    def info(self):
        s = self.session
        labels = labels_from_logits(s.logits.numpy())
        return {'case_id': s.case_id, 'shape': s.shape, 'split': s.split,
                'axis_order': 'native voxel [x,y,z]; display horizontal=x, vertical=y',
                'modality_mask': s.modality_mask.tolist(), 'rounds': len(s.history),
                'controller': self.controller_name, 'gt_available_to_model': False,
                'correction_modality': {'index': s.corrector.modality_index,
                                        'name': MODALITIES[s.corrector.modality_index]},
                'volumes_voxels': {'WT': int((labels > 0).sum()), 'TC': int((labels >= 2).sum()),
                                   'ET': int((labels == 3).sum())}, 'history': s.history}

    def apply(self, request):
        allowed = {'points', 'box', 'target', 'text'}
        if not isinstance(request, dict) or set(request) - allowed:
            raise ValueError('Unexpected manual prompt fields')
        points, box = request.get('points', []), request.get('box')
        if not isinstance(points, list) or len(points) > 32:
            raise ValueError('One round supports at most 32 registered points')
        target, text = request.get('target', 'ET'), request.get('text', '')
        if target not in REGIONS or not isinstance(text, str) or not 1 <= len(text) <= 2000:
            raise ValueError('Choose WT/TC/ET and supply an explicit include/exclude instruction')
        roi = bounded_roi(points, box, self.session.shape, self.roi_size)
        self.serial += 1
        roi_id = f'roi_{self.serial}'
        self.session.register_roi(roi_id, roi)
        ids = []
        for n, point in enumerate(points):
            value = PointPrompt(f'point_{self.serial}_{n}', tuple(point['voxel']), target, point['positive'])
            self.session.register_point(value)
            ids.append(value.point_id)
        box_id = None
        if box is not None:
            if not any(b - a == 1 for a, b in zip(box['start'], box['stop'])):
                raise ValueError('Draw a 2D slice box, not a 3D volume box')
            box_id = f'box_{self.serial}'
            self.session.register_box(BoxPrompt(box_id, tuple(box['start']), tuple(box['stop']), target, box['positive']))
        action = self.controller.propose(text, self.session, roi_id=roi_id, point_ids=tuple(ids),
                                         target=target, box_prompt=box is not None, box_id=box_id)
        result = self.session.apply_action(action)
        self.commands.append({'action': action.to_dict(), 'user_text': text})
        return {'action': action.to_dict(), 'result': result, 'info': self.info()}

    def undo(self):
        undone = self.session.undo()
        if undone:
            self.commands.append({'undo': True})
        return {'undone': undone, 'info': self.info()}

    def assign_class(self, request):
        """One explicitly class-labelled human point -> two existing region tools.

        This is a deterministic hierarchy contract, not a new ED/NCR model.
        Failure of either tool restores the state before both tools.
        """
        if not isinstance(request, dict) or set(request) - {'class_name', 'voxel'}:
            raise ValueError('Atomic assignment requires only explicit class_name and voxel')
        name, voxel = request.get('class_name'), request.get('voxel')
        if name not in ('ED', 'NCR'):
            raise ValueError('Choose explicit ED or NCR; a negative class point is insufficient')
        # Validate before any model call.
        bounded_roi([{'voxel': voxel}], None, self.session.shape, self.roi_size)
        mapping = {'ED': (('WT', True), ('TC', False)),
                   'NCR': (('TC', True), ('ET', False))}
        command_length = len(self.commands)
        results = []
        try:
            with atomic_point_transaction(self.session, voxel, name):
                for target, positive in mapping[name]:
                    text = f'{target}漏分了，补上这个正点附近的{target}' if positive else f'{target}分多了，去掉这个位置的{target}'
                    result = self.apply({'target': target, 'text': text,
                                         'points': [{'voxel': voxel, 'positive': positive}]})
                    results.append(result['action'])
        except Exception:
            del self.commands[command_length:]
            raise
        del self.commands[command_length:]
        self.commands.append({'atomic_assignment': {'class_name': name, 'voxel': voxel},
                              'actions': results, 'human_prompt_count': 1, 'visual_tool_calls': 2})
        return {'class_name': name, 'voxel': voxel, 'human_prompt_count': 1,
                'visual_tool_calls': 2, 'actions': results,
                'scope': 'explicit point-class constraint; neighboring voxels remain model predictions',
                'info': self.info()}

    def slice_png(self, z, modality, initial=False):
        if not 0 <= z < self.session.shape[2] or modality not in range(4):
            raise ValueError('Invalid slice/modality')
        image = self.raw[modality, :, :, z].T
        nonzero = image[image != 0]
        low, high = np.percentile(nonzero, (1, 99)) if nonzero.size else (0, 1)
        gray = np.clip((image - low) / max(float(high - low), 1e-6), 0, 1)
        rgb = np.repeat((gray * 255).astype(np.float32)[..., None], 3, axis=2)
        logits = self.session.initial_logits if initial else self.session.logits
        labels = labels_from_logits(logits[:, :, :, z:z + 1].numpy())[:, :, 0].T
        for value, color in ((1, (55, 190, 110)), (2, (80, 160, 255)), (3, (255, 80, 90))):
            selected = labels == value
            rgb[selected] = .62 * rgb[selected] + .38 * np.array(color)
        buffer = io.BytesIO()
        Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8)).save(buffer, format='PNG')
        return buffer.getvalue()

    def export(self):
        self.export_serial += 1
        destination = self.output / f'export_{self.export_serial:03d}'
        destination.mkdir(parents=True, exist_ok=False)
        s = self.session
        nib.save(nib.Nifti1Image(labels_from_logits(s.logits.numpy()), s.affine),
                 destination / 'corrected_internal_labels.nii.gz')
        replay = {'case_id': s.case_id, 'original_case': str(self.case_path.resolve()),
                  'original_case_sha256': digest(self.case_path),
                  'rois': {key: asdict(value) for key, value in s.rois.items()},
                  'points': [asdict(value) for value in s.points.values()],
                  'boxes': [asdict(value) for value in s.boxes.values()], 'commands': self.commands,
                  'prompt_source': 'human_browser', 'gt_loaded': False}
        write_json(destination / 'interactions.json', replay)
        write_json(destination / 'session.json', {**self.info(), 'model_parameters_updated': False,
                    'initialization': self.session.source_provenance, 'commands': self.commands})
        return {'saved': str(destination)}


def serve(args):
    if args.output.exists():
        raise FileExistsError('Choose a new output directory')
    session, controller, case = make_session(args)
    args.output.mkdir(parents=True)
    state = DemoState(session, controller, case['raw_mri'], args.output, case_path=args.case,
                      controller_name=args.controller, roi_size=args.roi_size)
    token = secrets.token_urlsafe(24)
    html = (Path(__file__).with_name('demo_ui.html')).read_bytes()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *unused):
            pass

        def respond(self, body, status=200, content_type='application/json; charset=utf-8'):
            self.send_response(status)
            self.send_header('Content-Type', content_type)
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.end_headers()
            self.wfile.write(body)

        def authenticated(self):
            return secrets.compare_digest(parse_qs(urlparse(self.path).query).get('token', [''])[0], token)

        def do_GET(self):
            if not self.authenticated():
                return self.respond(b'{"error":"invalid local session token"}', 403)
            path, query = urlparse(self.path).path, parse_qs(urlparse(self.path).query)
            try:
                if path == '/':
                    return self.respond(html, content_type='text/html; charset=utf-8')
                if path == '/info':
                    body = json.dumps(state.info(), ensure_ascii=False).encode()
                elif path == '/slice':
                    body = state.slice_png(int(query['z'][0]), int(query.get('modality', ['1'])[0]),
                                           query.get('initial', ['0'])[0] == '1')
                    return self.respond(body, content_type='image/png')
                else:
                    return self.respond(b'{"error":"not found"}', 404)
                self.respond(body)
            except Exception as error:
                self.respond(json.dumps({'error': str(error)}).encode(), 400)

        def do_POST(self):
            if not self.authenticated() or self.headers.get('X-Demo-Request') != 'manual':
                return self.respond(b'{"error":"invalid local session request"}', 403)
            try:
                length = int(self.headers.get('Content-Length', '0'))
                if not 0 < length < 65536:
                    raise ValueError('Invalid request length')
                body = json.loads(self.rfile.read(length))
                path = urlparse(self.path).path
                if path == '/apply':
                    result = state.apply(body)
                elif path == '/assign-class':
                    result = state.assign_class(body)
                elif path == '/undo':
                    result = state.undo()
                elif path == '/export':
                    result = state.export()
                else:
                    return self.respond(b'{"error":"not found"}', 404)
                self.respond(json.dumps(result, ensure_ascii=False).encode())
            except Exception as error:
                self.respond(json.dumps({'error': f'{type(error).__name__}: {error}'}, ensure_ascii=False).encode(), 400)

    server = HTTPServer(('127.0.0.1', args.port), Handler)
    print(f'Open http://127.0.0.1:{server.server_port}/?token={token}', flush=True)
    print('No GT loaded. Ctrl+C stops this local server. Export saves a new prediction.', flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()


def metrics(prediction, gt):
    rows = {}
    for name, pred, truth in zip((*REGIONS, 'ED', 'NCR'),
                                 (prediction > 0, prediction >= 2, prediction == 3,
                                  prediction == 1, prediction == 2),
                                 (gt > 0, gt >= 2, gt == 3, gt == 1, gt == 2)):
        tp = int((pred & truth).sum())
        fp, fn = int((pred & ~truth).sum()), int((~pred & truth).sum())
        denominator = 2 * tp + fp + fn
        rows[name] = {'dice': 2 * tp / denominator if denominator else None, 'tp': tp, 'fp': fp, 'fn': fn}
    return rows


def verified_gt(path, shape, affine):
    image = nib.load(str(path))
    if image.shape != shape or not np.allclose(image.affine, affine, atol=1e-5, rtol=0):
        raise ValueError('GT geometry differs from current native image grid')
    return internal_labels(image)


def score(args):
    prediction = nib.load(str(args.prediction))
    labels = internal_labels(prediction)
    gt = verified_gt(args.gt, prediction.shape, prediction.affine)
    report = {'prediction': str(args.prediction), 'gt_use': 'scoring_only', 'regions': metrics(labels, gt)}
    if args.baseline:
        baseline = nib.load(str(args.baseline))
        if baseline.shape != prediction.shape or not np.allclose(baseline.affine, prediction.affine, atol=1e-5, rtol=0):
            raise ValueError('Baseline geometry mismatch')
        report['baseline_regions'] = metrics(internal_labels(baseline), gt)
        for key in (*REGIONS, 'ED', 'NCR'):
            before, after = report['baseline_regions'][key]['dice'], report['regions'][key]['dice']
            report['regions'][key]['delta'] = None if before is None or after is None else after - before
    if args.output:
        if args.output.exists():
            raise FileExistsError(args.output)
        write_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def simulate(args):
    # Explicit exploratory one-case pilot. This does not replace frozen-protocol
    # test100 evaluation, and a positive result must not be presented as that.
    from .training import simulate_error_click
    if args.output.exists():
        raise FileExistsError('Choose a new output directory')
    session, controller, case = make_session(args)
    gt = verified_gt(args.gt, session.shape, session.affine)
    args.output.mkdir(parents=True)
    state = DemoState(session, controller, case['raw_mri'], args.output, case_path=args.case,
                      controller_name=args.controller, roi_size=args.roi_size)
    rows = [{'click': 0, 'regions': metrics(labels_from_logits(session.logits.numpy()), gt)}]
    for step in range(1, args.clicks + 1):
        point = simulate_error_click(torch.sigmoid(session.logits[2]).numpy(), gt == 3,
                                     'ET', session.affine, args.seed + step)
        if point is None:
            break
        request = {'target': 'ET', 'points': [{'voxel': list(point.voxel), 'positive': point.positive}],
                   'text': 'ET漏分了，补上这个正点附近的ET' if point.positive else 'ET分多了，保留周围水肿'}
        outcome = state.apply(request)
        prediction = labels_from_logits(session.logits.numpy())
        nib.save(nib.Nifti1Image(prediction, session.affine), args.output / f'click_{step:02d}.nii.gz')
        rows.append({'click': step, 'regions': metrics(prediction, gt), 'action': outcome['action']})
        print(json.dumps(rows[-1], ensure_ascii=False), flush=True)
    report = {'case_id': session.case_id, 'split': session.split, 'status': 'complete',
              'purpose': 'exploratory_single_case_pilot_not_blind_test100_result',
              'prompt_source': 'GT_error_simulator', 'gt_passed_to_model_or_llm': False,
              'model_parameters_updated': False, 'controller': args.controller,
              'initialization': case['metadata'], 'rows': rows,
              'commands': state.commands, 'history': session.history,
              'rois': {k: asdict(v) for k, v in session.rois.items()},
              'points': [asdict(v) for v in session.points.values()]}
    write_json(args.output / 'pilot_metrics.json', report)
    # Deliberately do not label a GT-generated sequence as human_browser.
    print(json.dumps({'saved': str(args.output), 'actual_clicks': len(rows) - 1}, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    check = commands.add_parser('preflight', help='Check packages, local model files and CUDA; load no weights')
    check.add_argument('--nninteractive-model', type=Path, default=DEFAULT_MODEL)
    check.add_argument('--llm-path', type=Path, default=DEFAULT_QWEN)
    check.add_argument('--device', default='cuda:0')
    check.add_argument('--require-qwen', action='store_true')
    check.add_argument('--output', type=Path)
    check.set_defaults(function=preflight)
    prep = commands.add_parser('prepare', help='Prepare a GT-free bundle from Legacy hard labels + raw MRI')
    prep.add_argument('--prediction', type=Path, required=True)
    for name in MODALITIES:
        prep.add_argument('--' + name.lower(), type=Path)
    prep.add_argument('--case-id', required=True)
    prep.add_argument('--split', choices=('train', 'validation', 'test', 'unassigned'), required=True)
    prep.add_argument('--output', type=Path, required=True)
    prep.set_defaults(function=prepare)
    for command, function in (('serve', serve), ('simulate', simulate)):
        sub = commands.add_parser(command)
        sub.add_argument('--case', type=Path, required=True)
        sub.add_argument('--nninteractive-model', type=Path, default=DEFAULT_MODEL)
        sub.add_argument('--modality', type=int, choices=range(4), default=1)
        sub.add_argument('--device', default='cuda:0')
        sub.add_argument('--controller', choices=('rule', 'qwen'), default='rule')
        sub.add_argument('--llm-path', type=Path, default=DEFAULT_QWEN)
        sub.add_argument('--llm-device', default='cuda:1')
        sub.add_argument('--roi-size', type=int, default=96)
        sub.add_argument('--output', type=Path, required=True)
        sub.set_defaults(function=function)
        if command == 'serve':
            sub.add_argument('--port', type=int, default=8765)
        else:
            sub.add_argument('--gt', type=Path, required=True)
            sub.add_argument('--clicks', type=int, default=3)
            sub.add_argument('--seed', type=int, default=0)
    scoring = commands.add_parser('score', help='Separate GT-only metric step; no inference')
    scoring.add_argument('--prediction', type=Path, required=True)
    scoring.add_argument('--baseline', type=Path)
    scoring.add_argument('--gt', type=Path, required=True)
    scoring.add_argument('--output', type=Path)
    scoring.set_defaults(function=score)
    args = parser.parse_args()
    if getattr(args, 'roi_size', 96) < 16 or getattr(args, 'clicks', 1) < 1:
        parser.error('ROI must be >=16 voxels and clicks >=1')
    torch.set_num_threads(4)
    result = args.function(args)
    if isinstance(result, int):
        raise SystemExit(result)
