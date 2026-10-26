"""Exercise actual validation through b2nd and enabled trainer diagnostics."""
import argparse
import copy
import json
import shutil
from pathlib import Path

import numpy as np
import torch

from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDatasetBlosc2, comp_blosc2_params
from nnunetv2.training.nnUNetTrainer.variants.brats2021_audit.nnUNetTrainer_BraTSModalityAgent300 import nnUNetTrainer_BraTSAgentLoMixLoss300
from .agent_evaluate import write_json
from .agent_integration_check import batch_for
from .missing_predictor import preprocess_available_case, MissingModalityPredictor
from .missing_modality import MODALITIES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixtures', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--dataset-root', type=Path, default=Path('/data02/chujingyi/data/nnUNet_preprocessed/Dataset137_BraTS2021'))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    plans = json.loads((args.dataset_root / 'nnUNetPlans.json').read_text()); plans['continue_training'] = False
    config = plans['configurations']['3d_fullres']; config['patch_size'] = [64, 64, 32]
    config['architecture']['arch_kwargs']['features_per_stage'] = [8, 16, 32, 64, 80, 80]
    dataset_json = json.loads((args.dataset_root / 'dataset.json').read_text())
    t = nnUNetTrainer_BraTSAgentLoMixLoss300(copy.deepcopy(plans), '3d_fullres', 0, dataset_json, device=torch.device('cuda'))
    t.initialize(); t.output_folder = str(args.output / 'trainer'); Path(t.output_folder).mkdir(exist_ok=True)
    t.current_epoch = 120; t.network.set_epoch(120); t._agent_diag_every = 1; t._agent_probe_every = 1
    batch = batch_for(t)
    train = t.train_step(batch); t.on_train_epoch_end([train])
    t.on_validation_epoch_start()
    with torch.no_grad():
        val = t.validation_step(batch); t.on_validation_epoch_end([val])
    records = [json.loads(line) for line in (Path(t.output_folder) / 'module_diagnostics.jsonl').read_text().splitlines()]
    kinds = {r['kind'] for r in records}
    assert {'AGENT_STEP', 'AGENT_ROUTING_BY_MASK', 'AGENT_LOSS', 'AGENT_15_MASK_PATCH_PROBE'}.issubset(kinds)
    probe = next(r for r in records if r['kind'] == 'AGENT_15_MASK_PATCH_PROBE')
    assert len(probe['masks']) == 15
    interventions = probe['masks']['15']['branch_interventions']['interventions']
    assert len(interventions) == 9
    assert 'lomix_weights' in next(r for r in records if r['kind'] == 'AGENT_STEP')
    modalities = {m: str(args.fixtures / 'synthetic_raw' / (m + '.nii.gz')) for m in MODALITIES}
    data, properties, mask = preprocess_available_case(modalities, t.plans_manager, t.configuration_manager, dataset_json)
    fake_root = args.output / 'preprocessed_fixture'; folder = fake_root / 'data'; folder.mkdir(parents=True, exist_ok=True)
    gt = fake_root / 'gt_segmentations'; gt.mkdir(exist_ok=True)
    shutil.copyfile(args.fixtures / 'synthetic_gt/synthetic.nii.gz', gt / 'synthetic.nii.gz')
    seg = np.zeros((1, *data.shape[1:]), dtype=np.int16)
    blocks, chunks = comp_blosc2_params(data.shape, (64, 64, 32), data.numpy().dtype.itemsize)
    sb, sc = comp_blosc2_params(seg.shape, (64, 64, 32), seg.dtype.itemsize)
    nnUNetDatasetBlosc2.save_case(data.numpy(), seg, properties, str(folder / 'synthetic'),
                                 chunks=chunks, blocks=blocks, chunks_seg=sc, blocks_seg=sb)
    t.preprocessed_dataset_folder_base, t.preprocessed_dataset_folder = str(fake_root), str(folder)
    t.dataset_class = nnUNetDatasetBlosc2
    t.do_split = lambda: ([], ['synthetic'])
    t.inference_allowed_mirroring_axes = (0,)
    summary = t.perform_actual_validation()
    assert summary['n_masks'] == 15 and t.network.decoder.deep_supervision and not t.network.training
    # Verify all 8 mirror variants explicitly carry the same one-modality mask.
    predictor = MissingModalityPredictor(device=t.device, perform_everything_on_device=False, use_mirroring=True)
    t.network.decoder.deep_supervision = False
    predictor.manual_initialization(t.network, t.plans_manager, t.configuration_manager, None, dataset_json, type(t).__name__, (0, 1, 2))
    output, routes = predictor.predict_with_mask(data, torch.tensor([0, 1, 0, 0]))
    assert len(routes) == 8 and all(r['mask_codes'] == [2] for r in routes)
    assert torch.isfinite(output).all() and predictor._case_mask is None
    write_json(args.output / 'report.json', {'status': 'passed', 'enabled_diagnostics': sorted(kinds),
        'probe_masks': 15, 'branch_interventions': len(interventions), 'actual_validation_b2nd_masks': summary['n_masks'], 'ds_and_training_mode_restored': True,
        'single_modality_all_eight_tta_variants': True, 'scope': 'synthetic integration; no research performance claims'})


if __name__ == '__main__':
    main()
