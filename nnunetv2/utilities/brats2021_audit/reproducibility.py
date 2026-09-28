"""Same initialization and batch-indexed augmentation across audit ablations."""
import hashlib
import os
import random

import numpy as np
import torch
from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter
from batchgenerators.dataloading.single_threaded_augmenter import SingleThreadedAugmenter


def configure_determinism():
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def seed_experiment(seed):
    if not 0 <= seed < 2**32:
        raise ValueError('BRATS_AUDIT_SEED must be in [0, 2**32)')
    # Must precede the first CUDA matrix multiplication.
    configure_determinism()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def state_digest(module):
    h = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        h.update(name.encode())
        h.update(str(tuple(tensor.shape)).encode())
        h.update(str(tensor.dtype).encode())
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


class SeededBatchLoader:
    """Seed Python/NumPy/CPU torch per *logical* batch, independent of workers.

    nnUNetDataLoader is an infinite sampler using global NumPy RNG. Its spatial
    transforms also use CPU torch RNG. Epoch-boundary resume starts at the same
    logical index without replaying earlier patches. Worker CUDA is never used.
    """
    def __init__(self, loader, seed, stream, workers, offset):
        self.loader, self.seed, self.stream = loader, int(seed), int(stream)
        self.workers, self.offset = max(1, workers), int(offset)
        self.thread_id, self.draw = 0, 0

    def set_thread_id(self, thread_id):
        self.thread_id, self.draw = thread_id, 0
        self.loader.set_thread_id(thread_id)

    def __iter__(self):
        return self

    def __next__(self):
        index = self.offset + self.thread_id + self.draw * self.workers
        seed = int(np.random.SeedSequence([self.seed, self.stream, index]).generate_state(1)[0])
        py_state, np_state, th_state = random.getstate(), np.random.get_state(), torch.get_rng_state()
        try:
            random.seed(seed)
            np.random.seed(seed)
            torch.random.default_generator.manual_seed(seed)
            batch = next(self.loader)
        finally:
            random.setstate(py_state)
            np.random.set_state(np_state)
            torch.set_rng_state(th_state)
        self.draw += 1
        return batch


def seeded_augmenter(loader, seed, stream, workers, offset, pin_memory):
    source = SeededBatchLoader(loader, seed, stream, workers, offset)
    if workers == 0:
        source.set_thread_id(0)
        return SingleThreadedAugmenter(source, None)
    seeds = [int(np.random.SeedSequence([seed, stream, i]).generate_state(1)[0]) for i in range(workers)]
    return MultiThreadedAugmenter(source, None, workers, num_cached_per_queue=2,
                                  seeds=seeds, pin_memory=pin_memory, wait_time=.002)
