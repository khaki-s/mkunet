import argparse
from pathlib import Path
import importlib.util
import json
import sys
import time
import torch

from .common import read_json, write_json, import_manifest
from . import pipeline
from .parallel import add_partition_arguments, merge_collections

ROOT = Path('/data02/chujingyi/brats_dataset_audit/foundation_autoprompt_20261113')


def parser():
    p = argparse.ArgumentParser(description='Frozen SAM3 -> PromptMoE utility router -> frozen nnInteractive')
    sub = p.add_subparsers(dest='command', required=True)
    q = sub.add_parser('init-config', help='Write explicit model/protocol paths; no inference')
    q.add_argument('--output', required=True)
    q.add_argument('--kind', choices=('point', 'box', 'point_box'), default='point')
    q.add_argument('--sam3-source', default='/data02/chujingyi/sam3-main')
    q.add_argument('--sam3-checkpoint', default='/data02/chujingyi/model/sam3/master/sam3.pt')
    q.add_argument('--router-source', default=str(ROOT / 'third_party/PromptMoE/PromptMoE.py'))
    q.add_argument('--correction-protocol', default='/data02/chujingyi/brats_dataset_audit/interactive_val80_protocol_20261105/protocol_local_correction_v2.json')
    q = sub.add_parser('preflight', help='Read paths/dependencies/CUDA; no model download')
    q.add_argument('--config', required=True)
    q.add_argument('--output')
    q.add_argument('--load-models', action='store_true', help='Actually load both official weights; requires compatible environment')
    q.add_argument('--device', default='cuda')
    q = sub.add_parser('import-manifest', help='Reuse verified MRI and nnUNet masks, no re-prediction')
    q.add_argument('--source', required=True)
    q.add_argument('--output', required=True)
    q = sub.add_parser('collect', help='train320 only: measure actual frozen-nnInteractive candidate gains')
    for option in ('manifest', 'config', 'output'):
        q.add_argument('--'+option, required=True)
    q.add_argument('--device', default='cuda')
    q.add_argument('--rounds', type=int, default=5)
    q.add_argument('--nninteractive-device', help='Optional second GPU for the frozen 3D corrector')
    q.add_argument('--cuda-stats', help='Write measured allocated/reserved peaks per visible device')
    add_partition_arguments(q)
    q.add_argument('--limit-cases', type=int)
    q.add_argument('--pairs', action='store_true')
    q.add_argument('--save-adapter-pairs', action='store_true')
    q.add_argument('--rollout-policy')
    q.add_argument('--rollout-margin-pp', type=float, default=0.1)
    q = sub.add_parser('merge-feedback', help='Verify and combine disjoint GPU collection shards')
    q.add_argument('--datasets', nargs='+', required=True)
    q.add_argument('--output', required=True)
    q = sub.add_parser('feedback-report', help='Inspect training candidate coverage before expensive training')
    q.add_argument('--dataset', required=True)
    q.add_argument('--output', required=True)
    q.add_argument('--margin-pp', type=float, default=0.1)
    q = sub.add_parser('train', help='Only train the upstream PairRouter; foundations are offline/frozen')
    q.add_argument('--datasets', required=True, nargs='+')
    q.add_argument('--output', required=True)
    q.add_argument('--device', default='cpu')
    q.add_argument('--epochs', type=int, default=100)
    q.add_argument('--batch-size', type=int, default=32)
    q.add_argument('--lr', type=float, default=1e-3)
    q.add_argument('--seed', type=int, default=20261113)
    q.add_argument('--threads', type=int, default=4)
    q = sub.add_parser('predict', help='No GT input; test100 requires a passing val80 lock')
    for option in ('manifest', 'config', 'checkpoint', 'output'):
        q.add_argument('--'+option, required=True)
    q.add_argument('--device', default='cuda')
    q.add_argument('--nninteractive-device', help='Optional second GPU for the frozen 3D corrector')
    q.add_argument('--cuda-stats', help='Write measured allocated/reserved peaks per visible device')
    q.add_argument('--margin-pp', type=float, default=0.1)
    q.add_argument('--limit-cases', type=int)
    q.add_argument('--pairs', action='store_true')
    q.add_argument('--validation-lock')
    q = sub.add_parser('score', help='Read completed predictions, then GT; never change a prediction')
    for option in ('manifest', 'predictions', 'output'):
        q.add_argument('--'+option, required=True)
    q = sub.add_parser('validation-lock', help='Promote a fixed method only after complete val80 improvement')
    q.add_argument('--summary', required=True)
    q.add_argument('--output', required=True)
    return p


def _dispatch(args):
    entry = getattr(pipeline, args.command.replace('-', '_'))
    if not getattr(args, 'cuda_stats', None):
        return entry(args)
    roles = {'SAM3': args.device, 'nnInteractive': args.nninteractive_device or args.device}
    devices = {}
    for role, name in roles.items():
        device = torch.device(name)
        if device.type != 'cuda':
            continue
        index = device.index if device.index is not None else torch.cuda.current_device()
        if index not in devices:
            free, total = torch.cuda.mem_get_info(index)
            torch.cuda.reset_peak_memory_stats(index)
            devices[index] = {'roles': [], 'name': torch.cuda.get_device_name(index),
                              'free_gib_before': free / 2**30, 'total_gib': total / 2**30}
        devices[index]['roles'].append(role)
    started = time.monotonic()
    status = 'failed'
    try:
        result = entry(args)
        for index in devices:
            torch.cuda.synchronize(index)
        status = 'completed'
        return result
    finally:
        for index, metrics in devices.items():
            metrics.update(peak_allocated_gib=torch.cuda.max_memory_allocated(index) / 2**30,
                           peak_reserved_gib=torch.cuda.max_memory_reserved(index) / 2**30)
        write_json(args.cuda_stats, {'status': status, 'seconds': time.monotonic() - started,
                                    'devices': devices, 'nninteractive_native_patch': [192] * 3})


def main(argv=None):
    args = parser().parse_args(argv)
    for name in ('rounds', 'limit_cases', 'epochs', 'batch_size', 'threads'):
        if getattr(args, name, None) is not None and getattr(args, name) < 1:
            raise ValueError(f'{name} must be positive')
    for name in ('margin_pp', 'rollout_margin_pp', 'lr'):
        value = getattr(args, name, None)
        if value is not None and (not float('-inf') < value < float('inf') or value < 0):
            raise ValueError(f'{name} must be finite and nonnegative')
    if args.command == 'init-config':
        if Path(args.output).exists():
            raise FileExistsError('Use a new config path')
        config = {'schema': 'foundation_prompt_config_v2', 'router_source': args.router_source,
            'sam3': {'source_dir': args.sam3_source, 'checkpoint': args.sam3_checkpoint,
                     'kind': args.kind, 'min_component_voxels': 3, 'slices_per_head': 1,
                     'box_padding': 2, 'box_max_extent': 64, 'proposal_margin': 16,
                     'repeat_radius_mm': 5.0, 'targets': ['WT', 'TC', 'ET']},
            'correction_protocol': read_json(args.correction_protocol)}
        write_json(args.output, config)
        print(args.output)
    elif args.command == 'preflight':
        from .sam3_candidates import validate_source
        config = read_json(args.config)
        result = validate_source(config['sam3']['source_dir'], config['sam3']['checkpoint'])
        result.update(cuda_available=torch.cuda.is_available(), torch_version=torch.__version__,
                      nninteractive_installed=importlib.util.find_spec('nnInteractive') is not None)
        folder = Path(config['correction_protocol']['model_folder'])
        result['nninteractive_checkpoint_exists'] = (folder / 'fold_0/checkpoint_final.pth').is_file()
        result['models_loaded'] = False
        if args.load_models:
            if str(args.device).startswith('cuda') and not torch.cuda.is_available():
                raise RuntimeError('CUDA is unavailable. No GPU inference was attempted.')
            candidate = pipeline.generator(config, args.device)
            from ..inference_runtime import new_session
            correction = config['correction_protocol'].get('correction', {})
            session = new_session(folder, args.device,
                cudnn_benchmark=correction.get('cudnn_benchmark', False))
            result['models_loaded'] = True
            result['sam3_frozen'] = all(not p.requires_grad for p in candidate.model.parameters())
            result['nninteractive_frozen'] = all(not p.requires_grad for p in session.network.parameters())
            result['nninteractive_patch_size'] = session.configuration_manager.patch_size
        if args.output:
            write_json(args.output, result)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0 if result['ready_for_import'] and result['nninteractive_checkpoint_exists'] else 2
    elif args.command == 'import-manifest':
        print(import_manifest(args.source, args.output))
    elif args.command == 'merge-feedback':
        print(merge_collections(args.datasets, args.output))
    else:
        print(_dispatch(args))
    return 0


if __name__ == '__main__':
    sys.exit(main())
