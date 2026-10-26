"""Synthetic checkpoint discovery, CLI resume flag, single-file export and TTA."""
import argparse
import copy
import gc
import json
import os
from pathlib import Path
import tempfile
import traceback

import nibabel as nib
import numpy as np
import torch

from ..agent_integration_check import batch_for
from ..missing_predictor import load_predictor, preprocess_available_case
from ..reproducibility import state_digest
from nnunetv2.training.nnUNetTrainer.variants.brats2021_audit.nnUNetTrainer_BraTSAgentV2Loss300 import nnUNetTrainer_BraTSAgentNullV2Loss300


def run(root, device, work):
    plans=json.loads((root/'nnUNetPlans.json').read_text())
    c=plans['configurations']['3d_fullres'];c['patch_size']=[64,64,32];c['batch_size']=2
    c['architecture']['arch_kwargs']['features_per_stage']=[8,16,32,64,80,80]
    data=json.loads((root/'dataset.json').read_text())
    os.environ['nnUNet_results']=str(work/'results')
    os.environ['nnUNet_wandb_enabled']='false'
    def make(resume):
        p=copy.deepcopy(plans);p['continue_training']=resume
        t=nnUNetTrainer_BraTSAgentNullV2Loss300(p,'3d_fullres',0,data,device=device)
        t.num_iterations_per_epoch=1;t.initialize()
        assert Path(t.output_folder).is_relative_to(work/'results')
        return t
    t=make(False);t.current_epoch=119;t._agent_steps=119;t.on_train_epoch_start()
    batch=batch_for(t);t.train_step(batch)
    t.inference_allowed_mirroring_axes=(0,)
    checkpoint=Path(t.output_folder)/'checkpoint_best.pth';t.save_checkpoint(str(checkpoint))
    resumed=make(True);resumed.load_checkpoint(checkpoint)
    assert resumed.logger.resume and resumed.current_epoch==120
    assert state_digest(t.network)==state_digest(resumed.network)
    t.current_epoch=120
    for trainer in (t,resumed):trainer.on_train_epoch_start();trainer.train_step(batch)
    assert state_digest(t.network)==state_digest(resumed.network)
    model=Path(t.output_folder).parent
    (model/'plans.json').write_text(json.dumps(t.plans_manager.plans))
    (model/'dataset.json').write_text(json.dumps(data))
    del t,resumed,batch;gc.collect()
    predictor=load_predictor(model,'checkpoint_best.pth',device=device,mirroring=True)
    shape=(28,32,24);affine=np.diag([-1.,1.,1.,1.]);affine[:3,3]=[23,-17,8]
    values=np.zeros(shape,np.float32)
    values[2:-2,3:-3,2:-2]=np.random.default_rng(3).uniform(1,20,(24,26,20))
    path=work/'only_T1ce.nii.gz';nib.save(nib.Nifti1Image(values,affine),path)
    case={'case_id':'synthetic_T1ce','modalities':{'T1ce':str(path)}}
    prefix=work/'prediction'/'synthetic_T1ce'
    routes,properties=predictor.predict_available_case(case,prefix)
    image=nib.load(str(prefix)+'.nii.gz')
    assert image.shape==shape and np.allclose(image.affine,affine,rtol=0,atol=1e-4)
    assert predictor._case_mask is None
    assert bool(predictor.network.communication_active) and int(predictor.network.stage_epoch)==119
    assert int(predictor.network.architecture_version)==2
    tensor,_,mask=preprocess_available_case(case['modalities'],predictor.plans_manager,
                                          predictor.configuration_manager,predictor.dataset_json)
    a,_=predictor.predict_with_mask(tensor,mask)
    hidden=tensor.clone();hidden[~mask]=1e5
    b,_=predictor.predict_with_mask(hidden,mask)
    torch.testing.assert_close(a,b,rtol=0,atol=0)
    return {'actual_cli_resume_flag_and_next_step_equal':True,'checkpoint_discovery_and_strict_load':True,
            'saved_stage_restored':119,'single_existing_T1ce_file':True,'geometry_preserved':True,
            'tta_axes':[0],'tta_window_forward_count':len(routes),'hidden_slots_invariant':True,
            'case_mask_released':True,'scope':'synthetic technical checks, no patient training or test inference'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-root',type=Path,default=Path('/data02/chujingyi/data/nnUNet_preprocessed/Dataset137_BraTS2021'))
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--device',choices=('cuda','cpu'),default='cpu')
    args=parser.parse_args();torch.set_num_threads(2)
    if args.device=='cuda':torch.cuda.set_per_process_memory_fraction(.5)
    report={'status':'running','device':args.device}
    try:
        with tempfile.TemporaryDirectory(prefix='brats_v2_runtime_') as work:
            report.update(run(args.dataset_root,torch.device(args.device),Path(work)))
        report['status']='passed'
    except BaseException:
        report['status']='failed';report['error']=traceback.format_exc();raise
    finally:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(report,indent=2)+'\n')
        print(f"{report['status']}: {args.output}")


if __name__=='__main__':main()
