"""Use the existing official nnInteractive adapter; never hard-clamp auto-points."""
import torch
from ..evaluation import make_backend, register_prompt, labels_to_logits, labels_from_logits
from ..inference_runtime import new_session, prediction_context
from ..session import CorrectionSession
from ..schemas import BoxPrompt
from .conv3d_workspace import install_depth_tiles


class FrozenCorrector:
    def __init__(self, mri, availability, affine, config, *, device='cuda', engine=None, backend=None):
        self.mri, self.availability, self.affine, self.config = mri, availability, affine, config
        correction = config.get('correction', {})
        tile_depth = config.get('conv3d_depth_tile')
        if (backend is None and engine is None and torch.device(device).type == 'cuda'
                and tile_depth is not None):
            # Configure the frozen network before its first CUDA forward.
            # Only the large Conv3D operations are tiled. Their full outputs
            # are assembled before the next operation, including normalization.
            benchmark = correction.get('cudnn_benchmark', False)
            engine = new_session(config['model_folder'], 'cpu', cudnn_benchmark=benchmark)
            engine.device = torch.device(device)
            install_depth_tiles(engine.network, tile_depth)
            engine.network.to(engine.device)
            with torch.cuda.device(engine.device), prediction_context(engine):
                if hasattr(engine, 'warmup'):
                    engine.warmup()
        self.backend = backend or make_backend(config, mri[config['modality_index']:config['modality_index']+1],
                                               affine, device=device, inference_session=engine)
        network = getattr(getattr(self.backend, 'engine', None), 'network', None)
        if isinstance(network, torch.nn.Module):
            network.eval().requires_grad_(False)

    def __call__(self, current, prompt, history):
        session = CorrectionSession(self.mri, self.availability, labels_to_logits(current),
                                    self.affine, self.backend, enforce_clicks=False)
        head = ('WT', 'TC', 'ET').index(prompt['head'])
        for number, previous in enumerate(history):
            if previous['head'] != prompt['head']:
                continue
            if previous['kind'] in ('point', 'point_box'):
                position = tuple(previous['voxel'])
                session.confirmed_mask[(head, *position)] = True
                session.confirmed_labels[(head, *position)] = float(previous['positive'])
            if previous['kind'] in ('box', 'point_box'):
                identifier = f'auto_box_{number}'
                session.register_box(BoxPrompt(identifier, tuple(previous['box_start']),
                                               tuple(previous['box_stop']), previous['head'], previous['positive']))
                session.history.append({'action': {'box_id': identifier}})
        action = register_prompt(session, prompt, len(history)+1, self.config['roi_size'])
        result = session.apply_action(action)
        result.update(prompt_source='SAM3_automatic', hard_click_constraints=False)
        return labels_from_logits(session.logits), result


def execute_plan(corrector, current, actions, indices, history):
    """Ordered positive-then-negative pair costs two prompts/tool calls.

    Every candidate is evaluated on an independent copy of the same starting
    state. The next chosen state is recomputed from new model predictions.
    """
    output = current.copy()
    local_history = list(history)
    records = []
    for index in indices:
        prompt = actions[index]
        if prompt is None:
            raise ValueError('Cannot execute an unavailable candidate')
        output, record = corrector(output, prompt, local_history)
        local_history.append(prompt)
        records.append(record)
    return output, local_history, records
