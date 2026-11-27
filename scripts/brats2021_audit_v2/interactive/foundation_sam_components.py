#!/usr/bin/env python3
"""Train and evaluate GT-supervised 3D SAM-to-nnInteractive prompts."""
import argparse
import importlib

import foundation_prompt


def add_shard(parser, *, gpu=False):
    parser.add_argument('--num-shards', type=int, default=1)
    parser.add_argument('--shard-index', type=int, default=0)
    if gpu:
        parser.add_argument('--device', default='cuda:0')


def add_paths(parser, *names):
    for name in names:
        parser.add_argument('--'+name, required=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='stage', required=True)
    q = sub.add_parser('cohort')
    q.add_argument('--train-manifest', required=True)
    q.add_argument('--val-manifest', required=True)
    q.add_argument('--output', required=True)
    q.add_argument('--train-cases', type=int, default=20)
    q.add_argument('--val-start', type=int, default=8)
    q.add_argument('--val-cases', type=int, default=5)
    q = sub.add_parser('dense-cache')
    q.add_argument('--cohort', required=True)
    q.add_argument('--output', required=True)
    q.add_argument('--reuse')
    add_shard(q, gpu=True)
    q = sub.add_parser('diagnose-dense')
    add_paths(q, 'cohort', 'dense', 'output')
    q = sub.add_parser('features')
    add_paths(q, 'cohort', 'dense', 'output')
    add_shard(q)
    q.add_argument('--split', required=True, choices=('train', 'validation'))
    q.add_argument('--profile', choices=('atlas', 'connected'), default='atlas')
    q.add_argument('--wait', type=int, default=0, help='Seconds to wait for upstream dense shards')
    q = sub.add_parser('direction')
    add_paths(q, 'cohort', 'features', 'output')
    q = sub.add_parser('feedback')
    add_paths(q, 'cohort', 'direction', 'config', 'output')
    add_shard(q, gpu=True)
    q = sub.add_parser('policy')
    add_paths(q, 'cohort', 'direction', 'feedback', 'config', 'output')
    q.add_argument('--routing', choices=('shared', 'independent-directions'), default='shared')
    q = sub.add_parser('predict')
    add_paths(q, 'cohort', 'direction', 'policy', 'features', 'config', 'output')
    add_shard(q, gpu=True)
    q = sub.add_parser('score')
    add_paths(q, 'cohort', 'manifest', 'predictions', 'policy', 'output')
    args = parser.parse_args(argv)
    if args.stage == 'cohort' and (args.train_cases < 1 or args.val_cases < 1 or args.val_start < 0):
        parser.error('Invalid cohort size or offset')
    name = 'sam_component_cache' if args.stage in ('cohort', 'dense-cache', 'diagnose-dense') else 'sam_component_pipeline'
    module = importlib.import_module(foundation_prompt.name + '.foundation_prompt_v2.sam_components.' + name)
    print(getattr(module, args.stage.replace('-', '_'))(args), flush=True)


if __name__ == '__main__':
    main()
