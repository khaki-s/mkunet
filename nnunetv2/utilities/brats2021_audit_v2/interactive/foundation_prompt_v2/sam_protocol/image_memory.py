"""Keep SAM3's unused language and grounding heads on CPU for image prompts."""
import fcntl
from itertools import chain
from pathlib import Path

import torch

from ..sam3_candidates import _load_official
from ..sam_dense import SAM_CHECKPOINT, SAM_SOURCE
from ..common import read_json, write_json
from . import background, dense
from .cohorts import rows


class _StreamedVisionBlock(torch.nn.Module):
    """Stage one frozen block at a time, with the original GPU forward and dtype."""

    def __init__(self, block):
        super().__init__()
        self.block = block

    def forward(self, image):
        state = {name: value.to(image.device) for name, value in
                 chain(self.block.named_parameters(), self.block.named_buffers())}
        return torch.func.functional_call(self.block, state, (image,), strict=True)


def load_image_only(device, *, stream_blocks=0):
    """Load the pinned checkpoint, then transfer precisely the image-prompt path."""
    device = torch.device(device)
    if device.type != 'cuda':
        raise ValueError('The image-only CUDA route requires a GPU')
    model, cpu_processor = _load_official(SAM_SOURCE, SAM_CHECKPOINT, 'cpu')
    vision = model.backbone.vision_backbone
    interactive = model.inst_interactive_predictor
    if stream_blocks:
        blocks = vision.trunk.blocks
        if not 1 <= stream_blocks <= len(blocks):
            raise ValueError(f'Choose between 1 and {len(blocks)} streamed vision blocks')
        start = len(blocks) - stream_blocks
        offloaded = list(blocks[start:])
        for index in range(start, len(blocks)):
            blocks[index] = torch.nn.Identity()
    vision.to(device)
    if stream_blocks:
        for index, block in enumerate(offloaded, start):
            blocks[index] = _StreamedVisionBlock(block).eval()
    interactive.to(device)
    processor = type(cpu_processor)(model, resolution=cpu_processor.resolution, device=str(device))
    return model, processor


def vision_factory(refiner, *, stream_blocks=0):
    def create(device):
        model, processor = load_image_only(device, stream_blocks=stream_blocks)
        return refiner(device, model=model, processor=processor)
    return create


def cache_background_vision(args):
    training = rows(args.cohort, 'train')
    validation = rows(args.cohort, 'validation')
    chosen = {'train': training, 'validation': validation, 'both': training + validation}[args.split]
    spec = read_json(args.cohort)
    contract = background.contract(read_json(spec['train_manifest']['path']))
    root = Path(args.output)
    stream_blocks = getattr(args, 'stream_blocks', 0)
    path = root / f'{args.split}_shard{args.shard_index}.json'
    root.mkdir(parents=True, exist_ok=True)
    with path.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        factory = vision_factory(background.BackgroundPairedRefiner, stream_blocks=stream_blocks)
        stats = dense.build_cache(chosen[args.shard_index::args.num_shards], args, root, contract, factory=factory)
        if stats['new_cases'] or not path.exists():
            write_json(path, {**stats, 'placement': 'vision_and_interactive_predictor', 'stream_blocks': stream_blocks})
    return path
