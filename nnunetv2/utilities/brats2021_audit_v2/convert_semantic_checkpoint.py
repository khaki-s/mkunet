"""Strict SemanticMoE -> modality-agent WEIGHTS migration (never a resume)."""
import argparse
import json
from pathlib import Path

import torch
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
from nnunetv2.utilities.find_objects import recursive_find_trainer_class_by_name
from .agent_evaluate import file_sha256
from .reproducibility import seed_experiment

NEW_KEYS = ('stage_epoch', 'communication_active', 'mix_coefficient')
NEW_PREFIXES = ('communication.', 'lomix.', 'semantic.beta_condition.')


def migrate_semantic_weights(network, checkpoint):
    if network.semantic is None:
        raise ValueError('Semantic migration requires a target with four experts')
    source = {k.removeprefix('module.'): v for k, v in checkpoint['network_weights'].items()}
    target = network.state_dict()
    missing, unexpected = set(target) - set(source), set(source) - set(target)
    allowed = lambda k: k in NEW_KEYS or k.startswith(NEW_PREFIXES)
    if unexpected or any(not allowed(k) for k in missing):
        raise ValueError(f'Incompatible SemanticMoE keys: missing={sorted(missing)}, unexpected={sorted(unexpected)}')
    if any(k in source for k in NEW_KEYS) or any(k.startswith(NEW_PREFIXES) for k in source):
        raise ValueError('Source is already an agent checkpoint; use matching resume instead')
    for k, value in source.items():
        if value.shape != target[k].shape:
            raise ValueError(f'Weight shape mismatch at {k}: {value.shape} vs {target[k].shape}')
        target[k] = value
    network.load_state_dict(target, strict=True)
    # Keep the old inference phase; communication residuals start at zero.
    network.set_epoch(50 if bool(source['semantic.fusion_active']) else 0)
    return {'copied_keys': len(source), 'new_keys': sorted(missing),
            'prediction_stage_epoch': int(network.stage_epoch), 'optimizer_migrated': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--target-trainer', default='nnUNetTrainer_BraTSAgentCommLoss300')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    state = torch.load(args.source, map_location='cpu', weights_only=False)
    if state['trainer_name'] not in ('nnUNetTrainer_BraTSSemanticMoELoss300', 'nnUNetTrainer_BraTSSemanticMoEUniformLoss300'):
        raise ValueError('Source must be an audited SemanticMoE300 checkpoint')
    init = state['init_args']
    plans = PlansManager(init['plans'])
    config = plans.get_configuration(init['configuration'])
    target_cls = recursive_find_trainer_class_by_name(args.target_trainer)
    if target_cls is None or not hasattr(target_cls, 'network_options'):
        raise ValueError('Unknown modality-agent target trainer')
    seed = int(state['network_weights']['initialization_seed'])
    seed_experiment(seed)
    network = target_cls.build_network_architecture(plans, config, 4, 3, True)
    report = migrate_semantic_weights(network, state)
    report.update(kind='semantic_weights_only', source=str(args.source.resolve()), source_sha256=file_sha256(args.source),
                  source_trainer=state['trainer_name'], source_epochs=state['current_epoch'],
                  target_trainer=args.target_trainer, target_options=target_cls.network_options(), seed=seed,
                  note='Additional training is pretraining + adaptation, not a fresh 300-epoch comparison')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'network_weights': network.state_dict(), 'migration': report}, args.output)
    args.output.with_suffix('.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
