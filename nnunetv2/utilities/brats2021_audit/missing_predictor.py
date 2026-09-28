"""Named available-file input, explicit per-case masks, sliding windows and TTA.

Missing files are never opened. Normalization/cropping use only available
images; outputs use the original nnU-Net geometry/export contract.
"""
import inspect
import itertools
import json
from pathlib import Path
import re
import threading

import nibabel as nib
import numpy as np
import torch
from acvl_utils.cropping_and_padding.padding import pad_nd_image

import nnunetv2
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.inference.sliding_window_prediction import compute_gaussian
from nnunetv2.inference.export_prediction import export_prediction_from_logits
from nnunetv2.preprocessing.preprocessors.default_preprocessor import DefaultPreprocessor
from nnunetv2.utilities.find_class_by_name import recursive_find_python_class
from .agent_diagnostics import output_summary
from .missing_modality import MODALITIES, validate_mask
from .reproducibility import configure_determinism


def canonical_modalities(modalities, base_dir=None):
    names = {m.lower(): m for m in MODALITIES}
    result = {}
    for name, filename in modalities.items():
        canonical = names.get(name.lower())
        if canonical is None or canonical in result:
            raise ValueError(f'Unknown or duplicate modality: {name}')
        path = Path(filename)
        if not path.is_absolute() and base_dir is not None:
            path = Path(base_dir) / path
        if not path.is_file():
            raise FileNotFoundError(path)
        result[canonical] = str(path.resolve())
    if not result:
        raise ValueError('At least one available image file is required')
    return result


def read_manifest(path):
    path = Path(path)
    document = json.loads(path.read_text())
    cases = document.get('cases', [document]) if isinstance(document, dict) else document
    if not isinstance(cases, list) or not cases:
        raise ValueError('Manifest must be a case object, list, or {"cases": [...]}')
    result, seen = [], set()
    for row in cases:
        key = row['case_id']
        if not isinstance(key, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', key) or key in seen:
            raise ValueError(f'Invalid or duplicate case_id: {key}')
        seen.add(key)
        result.append({'case_id': key, 'modalities': canonical_modalities(row['modalities'], path.parent)})
    return result


class AvailableModalityPreprocessor(DefaultPreprocessor):
    def __init__(self, mask, verbose=False):
        super().__init__(verbose)
        self.mask = validate_mask(torch.as_tensor(mask)[None], 1)[0].cpu().numpy()

    def run_case_npy(self, data, seg, properties, plans_manager, configuration_manager, dataset_json):
        data = np.array(data, dtype=np.float32, copy=True)
        if data.ndim != 4 or data.shape[0] != 4 or not np.isfinite(data).all():
            raise ValueError('Expected finite four-slot 3D data')
        data[~self.mask] = 0
        properties = dict(properties)
        properties['modality_mask'] = self.mask.astype(int).tolist()
        data, seg, properties = super().run_case_npy(data, seg, properties, plans_manager, configuration_manager, dataset_json)
        data[~self.mask] = 0
        if not np.isfinite(data).all():
            raise ValueError('Nonfinite preprocessed image')
        return data, seg, properties

    def _normalize(self, data, seg, configuration_manager, foreground_intensity_properties_per_channel):
        for c in range(4):
            if not self.mask[c]:
                data[c] = 0
                continue
            scheme = configuration_manager.normalization_schemes[c]
            cls = recursive_find_python_class(str(Path(nnunetv2.__path__[0]) / 'preprocessing/normalization'),
                                               scheme, 'nnunetv2.preprocessing.normalization')
            if cls is None:
                raise RuntimeError(f'Unknown normalization scheme {scheme}')
            normalizer = cls(use_mask_for_norm=configuration_manager.use_mask_for_norm[c],
                             intensityproperties=foreground_intensity_properties_per_channel[str(c)])
            data[c] = normalizer.run(data[c], seg[0])
        return data


def preprocess_available_case(modalities, plans_manager, configuration_manager, dataset_json):
    modalities = canonical_modalities(modalities)
    ordered = [(m, modalities[m]) for m in MODALITIES if m in modalities]
    reference = None
    for _, path in ordered:
        img = nib.load(path)
        if len(img.shape) != 3:
            raise ValueError(f'Expected a 3D NIfTI: {path}')
        if reference is not None and (img.shape != reference.shape or not np.allclose(img.affine, reference.affine, rtol=0, atol=1e-4)):
            raise ValueError(f'Available images have different shape/affine: {path}')
        if reference is None:
            reference = img
    data_available, properties = plans_manager.image_reader_writer_class().read_images([p for _, p in ordered])
    data = np.zeros((4, *data_available.shape[1:]), dtype=np.float32)
    mask = np.zeros(4, dtype=bool)
    for index, (name, _) in enumerate(ordered):
        slot = MODALITIES.index(name)
        data[slot], mask[slot] = data_available[index], True
    properties['available_files'] = modalities
    processed, _, properties = AvailableModalityPreprocessor(mask).run_case_npy(
        data, None, properties, plans_manager, configuration_manager, dataset_json)
    return torch.from_numpy(processed), properties, torch.from_numpy(mask)


class MissingModalityPredictor(nnUNetPredictor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._case_lock = threading.Lock()
        self._case_mask = None
        self._case_routes = []
        self._collect_routes = False
        self._case_id = None
        self._tile_index = 0
        self._fold_index = 0

    @torch.inference_mode()
    def _internal_maybe_mirror_and_predict(self, x):
        if self._case_mask is None:
            raise RuntimeError('Use predict_with_mask so every window receives explicit availability')
        mask = self._case_mask.to(x.device).expand(x.shape[0], -1)
        x = x.masked_fill(~mask[:, :, None, None, None], 0)
        axes = self.allowed_mirroring_axes if self.use_mirroring else None
        axes = tuple(axes or ())
        if len(set(axes)) != len(axes) or any(a not in (0, 1, 2) for a in axes):
            raise ValueError('Mirror axes must be distinct spatial axes 0/1/2')
        flips = [()] + [tuple(a + 2 for a in c) for size in range(1, len(axes) + 1) for c in itertools.combinations(axes, size)]
        total = None
        for tta_index, flip in enumerate(flips):
            inp = torch.flip(x, flip) if flip else x
            if self._supports_mask:
                parts = self.network(inp, return_aux=self._collect_routes, modality_mask=mask)
                if self._collect_routes:
                    output = parts['output']
                    self._case_routes.append({'case_id': self._case_id, 'fold_index': self._fold_index,
                                              'tile': self._tile_index, 'tta': tta_index, **output_summary(parts)})
                else:
                    output = parts
            else:
                output = self.network(inp)
            if isinstance(output, (list, tuple)):
                output = output[0]
            prediction = torch.flip(output, flip) if flip else output
            total = prediction.float() if total is None else total + prediction.float()
        return total / len(flips)

    @torch.inference_mode()
    def predict_sliding_window_return_logits(self, input_image):
        if self._case_mask is None:
            raise RuntimeError('Call predict_with_mask, not an unmasked sliding-window entry point')
        if input_image.ndim != 4 or input_image.shape[0] != 4:
            raise ValueError('Expected C=4 preprocessed 3D tensor')
        self.network.to(self.device).eval()
        data, slicer = pad_nd_image(input_image, tuple(self.configuration_manager.patch_size),
                                    'constant', {'value': 0}, True, None)
        result_device = self.device if self.perform_everything_on_device else torch.device('cpu')
        logits = torch.zeros((3, *data.shape[1:]), device=result_device, dtype=torch.float32)
        counts = torch.zeros(data.shape[1:], device=result_device, dtype=torch.float32)
        gaussian = (compute_gaussian(tuple(self.configuration_manager.patch_size), sigma_scale=1/8,
                                     value_scaling_factor=1, dtype=torch.float32, device=result_device)
                    if self.use_gaussian else torch.ones(tuple(self.configuration_manager.patch_size), device=result_device))
        for i, window in enumerate(self._internal_get_sliding_window_slicers(data.shape[1:])):
            self._tile_index = i
            patch = data[window][None].contiguous().to(self.device)
            with torch.autocast(device_type=self.device.type, enabled=self.device.type == 'cuda'):
                prediction = self._internal_maybe_mirror_and_predict(patch)[0].to(result_device)
            logits[window] += prediction * gaussian
            counts[window[1:]] += gaussian
        logits /= counts
        if not torch.isfinite(logits).all():
            raise FloatingPointError('Nonfinite sliding-window prediction')
        return logits[(slice(None), *slicer[1:])]

    @torch.inference_mode()
    def predict_with_mask(self, data, mask, *, case_id='case', collect_routes=True):
        """One case at a time; lock prevents cross-case mask or routing leakage."""
        if not self._case_lock.acquire(blocking=False):
            raise RuntimeError('Concurrent cases need separate predictor instances')
        try:
            mask = torch.as_tensor(mask)
            self._case_mask = validate_mask(mask[None] if mask.ndim == 1 else mask, 1)
            self._case_id, self._case_routes, self._collect_routes = case_id, [], bool(collect_routes)
            self._supports_mask = 'modality_mask' in inspect.signature(self.network.forward).parameters
            configure_determinism()
            total = None
            parameters = self.list_of_parameters if self.list_of_parameters is not None else [None]
            if not parameters:
                raise ValueError('No checkpoint parameters supplied')
            for i, weights in enumerate(parameters):
                self._fold_index = i
                if weights is not None:
                    self.network.load_state_dict(weights, strict=True)
                    if hasattr(self.network, 'restore_stage'):
                        self.network.restore_stage()
                logits = self.predict_sliding_window_return_logits(data).cpu()
                total = logits if total is None else total + logits
            return total / len(parameters), list(self._case_routes)
        finally:
            self._case_mask, self._case_id, self._case_routes = None, None, []
            self._case_lock.release()

    def predict_available_case(self, case, output_prefix, *, save_probabilities=False, collect_routes=True):
        data, properties, mask = preprocess_available_case(case['modalities'], self.plans_manager,
                                                           self.configuration_manager, self.dataset_json)
        logits, routes = self.predict_with_mask(data, mask, case_id=case['case_id'], collect_routes=collect_routes)
        output_prefix = Path(output_prefix)
        output_prefix.parent.mkdir(parents=True, exist_ok=True)
        export_prediction_from_logits(logits, properties, self.configuration_manager, self.plans_manager,
                                      self.dataset_json, str(output_prefix), save_probabilities, num_threads_torch=4)
        return routes, properties


def load_predictor(model_folder, checkpoint='checkpoint_final.pth', *, folds=(0,), device=torch.device('cuda'),
                   mirroring=True, tile_step=.5):
    folder = Path(model_folder)
    if folder.name.startswith('fold_'):
        folds = (folder.name.removeprefix('fold_'),)
        folder = folder.parent
    predictor = MissingModalityPredictor(device=device, tile_step_size=tile_step, use_gaussian=True,
                                         use_mirroring=mirroring, perform_everything_on_device=False,
                                         allow_tqdm=False)
    predictor.initialize_from_trained_model_folder(str(folder), folds, checkpoint)
    configure_determinism()
    return predictor
