#!/usr/bin/env python3
"""Frozen SAM3 prompt ablation and gated BraTS multi-modal evaluation."""
import argparse
import importlib
import sys

import foundation_prompt


def paths(parser, *keys):
    for key in keys:
        parser.add_argument('--' + key, required=True)


def shard(parser, *, gpu=False):
    parser.add_argument('--num-shards', type=int, default=1)
    parser.add_argument('--shard-index', type=int, default=0)
    if gpu:
        parser.add_argument('--device', default='cuda:0')


def prepare_sam_import(args):
    """Import the frozen image model on Volta using FP16 for CUDA autocast declarations."""
    if args.stage not in ('cache', 'cache-background', 'cache-background-vision', 'cache-formal',
                           'targeted-cache', 'expert-sam', 'appearance-sam', 'test-predict'):
        return
    if args.stage == 'test-predict' and args.phase not in ('sam', 'all'):
        return
    import torch

    device = torch.device(args.device)
    if device.type != 'cuda' or torch.cuda.get_device_capability(device)[0] >= 8 or 'sam3' in sys.modules:
        return
    repo = importlib.import_module(foundation_prompt.name + '.foundation_prompt_v2.sam_dense').SAM_SOURCE
    autocast = torch.autocast

    def compatible_autocast(device_type, dtype=None, enabled=True, cache_enabled=None):
        if device_type == 'cuda' and dtype == torch.bfloat16:
            dtype = torch.float16
        return autocast(device_type, dtype=dtype, enabled=enabled, cache_enabled=cache_enabled)

    sys.path.insert(0, repo)
    torch.autocast = compatible_autocast
    try:
        importlib.import_module('sam3.model_builder')
        importlib.import_module('sam3.model.sam3_image_processor')
    finally:
        torch.autocast = autocast
        sys.path.remove(repo)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='stage', required=True)
    p = sub.add_parser('prepare')
    paths(p, 'root', 'output')
    p.add_argument('--study', choices=('pilot', 'formal'), required=True)
    p.add_argument('--pilot-permit')
    p.add_argument('--val-offset', type=int, help='Optional frozen validation block for a new pilot')
    p.add_argument('--train-list', help='Optional JSON list of 100 patients from the original training fold')
    p = sub.add_parser('reuse-pilot-cache')
    paths(p, 'cohort', 'run-dir')
    p = sub.add_parser('cache')
    paths(p, 'cohort', 'output')
    p.add_argument('--split', choices=('train', 'validation', 'both'), default='both')
    shard(p, gpu=True)
    p = sub.add_parser('cache-background')
    paths(p, 'cohort', 'output')
    p.add_argument('--split', choices=('train', 'validation', 'both'), default='both')
    shard(p, gpu=True)
    p = sub.add_parser('cache-background-vision')
    paths(p, 'cohort', 'output')
    p.add_argument('--split', choices=('train', 'validation', 'both'), default='both')
    p.add_argument('--stream-blocks', type=int, default=0)
    shard(p, gpu=True)
    p = sub.add_parser('cache-formal')
    paths(p, 'cohort', 'output')
    p.add_argument('--vision-only', action='store_true')
    shard(p, gpu=True)
    p = sub.add_parser('targeted-cache')
    paths(p, 'cohort', 'previous', 'output')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--vision-only', action='store_true')
    p = sub.add_parser('targeted-score')
    paths(p, 'cohort', 'previous', 'raw', 'output')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--min-free-mib', type=int, default=8192)
    p = sub.add_parser('fit-critic')
    paths(p, 'cohort', 'samples', 'output')
    p.add_argument('--threads', type=int, default=4)
    p = sub.add_parser('assess-critic')
    paths(p, 'cohort', 'dense', 'responses', 'model', 'output')
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--execute', action='store_true')
    shard(p, gpu=True)
    p = sub.add_parser('group-sample')
    paths(p, 'cohort', 'dense', 'responses', 'output')
    p.add_argument('--group-mm', type=float, choices=(4., 8., 12.), default=8.)
    shard(p)
    p = sub.add_parser('group-fit')
    paths(p, 'cohort', 'samples', 'output')
    p.add_argument('--group-mm', type=float, choices=(4., 8., 12.), default=8.)
    p.add_argument('--threads', type=int, default=4)
    p = sub.add_parser('group-predict')
    paths(p, 'cohort', 'dense', 'responses', 'model', 'output')
    p.add_argument('--threads', type=int, default=4)
    shard(p)
    p = sub.add_parser('group-score')
    paths(p, 'cohort', 'truth', 'model', 'predictions', 'output')
    p = sub.add_parser('consensus-predict')
    paths(p, 'cohort', 'dense', 'responses', 'samples', 'model', 'output')
    p.add_argument('--device', default='cuda:0')
    p = sub.add_parser('consensus-score')
    paths(p, 'cohort', 'truth', 'predictions', 'output')
    p = sub.add_parser('expert-sam')
    paths(p, 'cohort', 'experts', 'output')
    p.add_argument('--split', choices=('train', 'validation'), required=True)
    p.add_argument('--stream-blocks', type=int, default=0)
    p.add_argument('--profile', choices=('expert_disagreement_v7', 'expert_multislice_v8'),
                   default='expert_disagreement_v7')
    p.add_argument('--ready-only', action='store_true', help='Cache currently completed independent inputs')
    shard(p, gpu=True)
    p = sub.add_parser('appearance-sam')
    paths(p, 'cohort', 'proposals', 'proposal-model', 'output')
    p.add_argument('--split', choices=('train', 'validation'), required=True)
    p.add_argument('--case-list', help='Optional frozen subset of training patients for calibration')
    p.add_argument('--stream-blocks', type=int, default=16)
    shard(p, gpu=True)
    p = sub.add_parser('expert-expand')
    paths(p, 'cohort', 'dense', 'experts', 'output')
    p.add_argument('--split', choices=('train', 'validation'), required=True)
    shard(p)
    p = sub.add_parser('expert-modality-expand')
    paths(p, 'cohort', 'dense', 'archive', 'output')
    p.add_argument('--split', choices=('train', 'validation'), default='validation')
    shard(p)
    p = sub.add_parser('expert-responses')
    paths(p, 'cohort', 'dense', 'output')
    p.add_argument('--split', choices=('train', 'validation'), required=True)
    p.add_argument('--case-list', help='Optional frozen subset of training patients for calibration')
    p.add_argument('--min-free-mib', type=int, default=5376)
    p.add_argument('--seed-from-sam', action='store_true')
    p.add_argument('--native-margin', type=float, choices=(0., 1., 2., 3.))
    shard(p, gpu=True)
    p = sub.add_parser('expert-probe-score')
    paths(p, 'cohort', 'truth', 'dense', 'responses', 'output')
    p = sub.add_parser('expert-anchor-responses')
    paths(p, 'cohort', 'dense', 'responses', 'output')
    p.add_argument('--split', choices=('train', 'validation'), required=True)
    p.add_argument('--min-free-mib', type=int, default=5200)
    shard(p, gpu=True)
    p = sub.add_parser('collect-expert-features')
    paths(p, 'cohort', 'dense', 'responses', 'output')
    p.add_argument('--split', choices=('train', 'validation'), required=True)
    p.add_argument('--min-free-mib', type=int, default=5200)
    shard(p, gpu=True)
    p = sub.add_parser('respond')
    paths(p, 'cohort', 'dense', 'output')
    p.add_argument('--min-free-mib', type=int, default=5376)
    shard(p, gpu=True)
    p = sub.add_parser('sample')
    paths(p, 'cohort', 'dense', 'responses', 'output')
    p.add_argument('--features')
    shard(p)
    p = sub.add_parser('fit')
    paths(p, 'cohort', 'dense', 'responses', 'samples', 'output')
    p.add_argument('--head', type=int, choices=(0, 1, 2), required=True)
    p.add_argument('--threads', type=int, default=8)
    p = sub.add_parser('fit-gpu')
    paths(p, 'cohort', 'dense', 'responses', 'samples', 'output')
    p.add_argument('--head', type=int, choices=(0, 1, 2), required=True)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--dice-patient-weighting', action='store_true')
    p.add_argument('--features')
    p = sub.add_parser('predict')
    paths(p, 'cohort', 'dense', 'responses', 'model', 'output')
    p.add_argument('--features')
    shard(p)
    p = sub.add_parser('prepare-spatial')
    paths(p, 'cohort', 'dense', 'responses', 'output')
    p.add_argument('--split', choices=('train', 'validation'), required=True)
    shard(p)
    p = sub.add_parser('fit-spatial')
    paths(p, 'cohort', 'volumes', 'output')
    p.add_argument('--head', type=int, choices=(0, 1, 2), required=True)
    p.add_argument('--epochs', type=int, default=6)
    p.add_argument('--device', default='cuda:0')
    p = sub.add_parser('predict-spatial')
    paths(p, 'cohort', 'volumes', 'model', 'output')
    shard(p, gpu=True)
    p = sub.add_parser('prepare-shape')
    paths(p, 'cohort', 'volumes', 'output')
    p.add_argument('--split', choices=('train', 'validation'), required=True)
    p.add_argument('--head', type=int, choices=(0, 1, 2))
    shard(p)
    p = sub.add_parser('fit-shape')
    paths(p, 'cohort', 'volumes', 'distances', 'output')
    p.add_argument('--head', type=int, choices=(0, 1, 2), required=True)
    p.add_argument('--epochs', type=int, default=6)
    p.add_argument('--device', default='cuda:0')
    p = sub.add_parser('predict-shape')
    paths(p, 'cohort', 'volumes', 'distances', 'model', 'output')
    shard(p, gpu=True)
    p = sub.add_parser('collect-probe')
    paths(p, 'cohort', 'volumes', 'distances', 'output')
    p.add_argument('--split', choices=('train', 'validation'), required=True)
    shard(p, gpu=True)
    p = sub.add_parser('fit-probe')
    paths(p, 'cohort', 'features', 'output')
    p.add_argument('--epochs', type=int, default=12)
    p.add_argument('--device', default='cuda:0')
    p = sub.add_parser('predict-probe')
    paths(p, 'cohort', 'features', 'model', 'output')
    p.add_argument('--device', default='cuda:0')
    p = sub.add_parser('score-probe')
    paths(p, 'cohort', 'truth', 'model', 'predictions', 'output')
    p = sub.add_parser('score-validation')
    paths(p, 'cohort', 'truth', 'model', 'predictions', 'dense', 'responses', 'output')
    p = sub.add_parser('admit')
    paths(p, 'score', 'output')
    p = sub.add_parser('permit-test')
    paths(p, 'cohort', 'model', 'score', 'output')
    p = sub.add_parser('prepare-test')
    paths(p, 'manifest', 'baselines', 'permit', 'output')
    p = sub.add_parser('test-predict')
    paths(p, 'matrix', 'permit', 'model', 'output')
    p.add_argument('--mask-code', type=int, choices=range(1, 16), required=True)
    p.add_argument('--phase', choices=('sam', 'interactive', 'export', 'all'), default='all')
    p.add_argument('--vision-only', action='store_true')
    p.add_argument('--min-free-mib', type=int, default=8192)
    shard(p, gpu=True)
    p = sub.add_parser('test-score')
    paths(p, 'matrix', 'permit', 'truth', 'output')
    p.add_argument('--mask-code', type=int, choices=range(1, 16), required=True)
    p = sub.add_parser('test-summary')
    paths(p, 'matrix', 'permit', 'output')
    args = parser.parse_args(argv)
    if hasattr(args, 'shard_index') and not 0 <= args.shard_index < args.num_shards:
        parser.error('Invalid shard index')
    prepare_sam_import(args)
    modules = {'prepare': 'cohorts', 'reuse-pilot-cache': 'cohorts', 'cache': 'dense',
               'cache-background': 'background', 'cache-formal': 'cohorts', 'respond': 'candidates',
               'cache-background-vision': 'image_memory',
               'targeted-cache': 'targeted_background', 'targeted-score': 'targeted_background',
               'fit-critic': 'error_critic', 'assess-critic': 'error_critic',
               'group-sample': 'grouped_readout', 'group-fit': 'grouped_readout',
               'group-predict': 'grouped_readout', 'group-score': 'grouped_readout',
               'consensus-predict': 'paired_consensus', 'consensus-score': 'paired_consensus',
               'expert-sam': 'expert_prompts', 'appearance-sam': 'appearance_prompts',
               'expert-expand': 'expert_prompts',
               'expert-modality-expand': 'expert_modality',
               'expert-responses': 'expert_prompts', 'expert-probe-score': 'expert_prompts',
               'expert-anchor-responses': 'expert_anchor',
               'collect-expert-features': 'expert_features',
               'sample': 'residual', 'fit': 'residual', 'fit-gpu': 'gpu_fusion', 'predict': 'residual',
               'prepare-spatial': 'spatial', 'fit-spatial': 'spatial', 'predict-spatial': 'spatial',
               'prepare-shape': 'shape_fusion', 'fit-shape': 'shape_fusion', 'predict-shape': 'shape_fusion',
               'collect-probe': 'interactive_probe', 'fit-probe': 'interactive_probe',
               'predict-probe': 'interactive_probe', 'score-probe': 'interactive_probe',
               'score-validation': 'scoring', 'admit': 'scoring', 'permit-test': 'scoring',
               'prepare-test': 'evaluation_matrix', 'test-predict': 'evaluation_matrix',
               'test-score': 'evaluation_matrix', 'test-summary': 'evaluation_matrix'}
    module = importlib.import_module(foundation_prompt.name + '.foundation_prompt_v2.sam_protocol.' + modules[args.stage])
    print(getattr(module, args.stage.replace('-', '_'))(args), flush=True)


if __name__ == '__main__':
    main()
