"""Memory controls for frozen nnInteractive; never modify installed packages."""
from contextlib import contextmanager
import inspect

import torch

NATIVE_PATCH_SIZE = (192, 192, 192)


def new_session(model_folder, device, *, cudnn_benchmark=False):
    from nnInteractive.inference.inference_session import nnInteractiveInferenceSession
    options = dict(device=torch.device(device), use_torch_compile=False,
                   verbose=False, torch_n_threads=4, do_autozoom=True)
    # Older upstream releases do not expose this option. Our callers rebuild
    # interactions from the current state and never use upstream undo.
    if 'enable_undo' in inspect.signature(nnInteractiveInferenceSession).parameters:
        options['enable_undo'] = False
    engine = nnInteractiveInferenceSession(**options)
    # Recent upstream loaders unconditionally call self.warmup() at the END of
    # initialize_from_trained_model_folder. That runs the ORIGINAL 192³ patch
    # with benchmark=True before our memory/eval policy is set, and can OOM
    # before the first case. Keep the trained patch, defer only this instance's
    # warmup, and never patch the installed class.
    warmup = getattr(engine, 'warmup', None)
    if warmup is not None:
        engine.warmup = lambda: None
    try:
        engine.initialize_from_trained_model_folder(str(model_folder))
    finally:
        if warmup is not None:
            del engine.warmup
    configure_session(engine, cudnn_benchmark=cudnn_benchmark)
    if warmup is not None:
        with prediction_context(engine):
            warmup()
    return engine


def configure_session(engine, *, cudnn_benchmark=False):
    if type(cudnn_benchmark) is not bool:
        raise ValueError('cudnn_benchmark must be boolean')
    manager = getattr(engine, 'configuration_manager', None)
    if manager is not None:
        patch = getattr(manager, 'patch_size', None)
        if patch is None:
            patch = manager.configuration['patch_size']
        if tuple(patch) != NATIVE_PATCH_SIZE:
            raise ValueError('This workflow requires the original checkpoint-native 192³ patch; '
                             'a different network patch is not supported')
    if manager is not None or hasattr(engine, '__dict__'):
        engine._brats_cudnn_benchmark = cudnn_benchmark
    network = getattr(engine, 'network', None)
    if isinstance(network, torch.nn.Module):
        network.eval().requires_grad_(False)


@contextmanager
def prediction_context(engine):
    """Upstream __init__ enables benchmarking; override during every forward.

    Restoring the previous setting avoids changing unrelated SAM/model callers.
    This removes autotuning workspace spikes, not the model's activation cost.
    """
    previous = torch.backends.cudnn.benchmark
    torch.backends.cudnn.benchmark = getattr(engine, '_brats_cudnn_benchmark', False)
    try:
        with torch.inference_mode():
            yield
    except torch.OutOfMemoryError as error:
        manager = getattr(engine, 'configuration_manager', None)
        patch = getattr(manager, 'patch_size', 'unknown')
        raise torch.OutOfMemoryError(
            f'{error}\nFrozen nnInteractive inference OOM: patch_size={patch}. '
            'Keep the native 192³ patch and use a GPU with sufficient free memory. '
            'The dense Atlas route does not load SAM3; point experiments can place '
            'SAM3 and nnInteractive on separate devices. '
            'Multiple GPUs do not pool memory for this single-volume forward.') from error
    finally:
        torch.backends.cudnn.benchmark = previous
