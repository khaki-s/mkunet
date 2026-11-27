#!/usr/bin/env python
"""Synthetic architecture, optimization, resume and export checks; no patients."""
import argparse
from copy import deepcopy
import gc
import json
import os
from pathlib import Path
import tempfile
import traceback

import nibabel as nib
import numpy as np
import torch

import nnunetv2
from nnunetv2.utilities.find_class_by_name import recursive_find_python_class
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager, ConfigurationManager
from nnunetv2.training.nnUNetTrainer.variants.brats2021_audit.nnUNetTrainer_BraTSAgentGeneralistExpertsOnlyLegacyV2Loss300 import (
    nnUNetTrainer_BraTSAgentGeneralistExpertsOnlyLegacyV2Loss300 as Trainer,
)
from nnunetv2.utilities.brats2021_audit_v2.agent_v1.blocks import FeatureUNet
from nnunetv2.utilities.brats2021_audit_v2.agent_v1.diagnostics import output_summary
from nnunetv2.utilities.brats2021_audit_v2.agent_evaluate import aggregate_case_routes
from nnunetv2.utilities.brats2021_audit_v2.agent_v2.experts_only.diagnostics import experts_only_summary
from nnunetv2.utilities.brats2021_audit_v2.missing_modality import mask_from_code
from nnunetv2.utilities.brats2021_audit_v2.missing_predictor import load_predictor
from nnunetv2.utilities.brats2021_audit_v2.reproducibility import state_digest


def counts(model):
    return dict(total=sum(p.numel() for p in model.parameters()),
                trainable=sum(p.numel() for p in model.parameters() if p.requires_grad),
                segmentation_UNets=sum(isinstance(m, FeatureUNet) for m in model.modules()),
                independent_experts=len(model.semantic.experts),
                beta_parameters=[n for n,p in model.named_parameters() if 'beta' in n],
                has_joint_encoder=hasattr(model,'encoder'),has_joint_decoder=hasattr(model,'decoder'))


def batch_for(trainer, batch_size=2):
    shape=trainer.configuration_manager.patch_size
    x=torch.randn(batch_size,4,*shape,device=trainer.device)
    labels=torch.randint(0,4,(batch_size,*shape),device=trainer.device)
    y=torch.stack([labels>i for i in range(3)],1).float()
    return {'data':x,'target':y}


def make(plans,data,*,resume=False,device=torch.device('cpu')):
    p=deepcopy(plans);p['continue_training']=resume
    trainer=Trainer(p,'3d_fullres',0,data,device=device)
    trainer.num_iterations_per_epoch=1
    trainer.initialize()
    return trainer


def architecture_checks(plans,report):
    manager=PlansManager(deepcopy(plans));actual=manager.get_configuration('3d_fullres')
    model=Trainer.build_network_architecture(manager,actual,4,3,False)
    report['actual_plan_parameters']=counts(model)
    assert report['actual_plan_parameters']['total']==9796853
    assert report['actual_plan_parameters']['segmentation_UNets']==5
    assert not report['actual_plan_parameters']['has_joint_encoder']
    assert not report['actual_plan_parameters']['has_joint_decoder']
    assert not report['actual_plan_parameters']['beta_parameters']
    branch_ids=[{id(p) for p in e.parameters()} for e in model.semantic.experts]
    assert all(not(a&b) for i,a in enumerate(branch_ids) for b in branch_ids[i+1:])
    assert not any('encoder.' in n or 'decoder.' in n or 'beta' in n for n in model.state_dict())
    assert len(list(model.parameters()))==len({id(p) for p in model.parameters()})
    del model;gc.collect()
    config=deepcopy(actual.configuration);config['patch_size']=[32,64,16]
    small=ConfigurationManager(config)
    model=Trainer.build_network_architecture(manager,small,4,3,False)
    x=torch.randn(1,4,*small.patch_size)
    for epoch,active,comm in [(0,False,False),(49,False,False),(50,True,False),(99,True,False),(100,True,True),(119,True,True)]:
        model.set_epoch(epoch)
        assert bool(model.semantic.fusion_active)==active and bool(model.communication_active)==comm
        assert all(p.requires_grad==active for p in model.semantic.router.parameters())
        assert not any(p.requires_grad for p in model.communication.gate.parameters())
    model.eval()
    model.set_epoch(0)
    with torch.inference_mode():
        p=model(x,True)
        assert p['output'].shape==(1,3,*small.patch_size)
        torch.testing.assert_close(p['final'],sum(p['local_experts'])/5,rtol=1e-6,atol=1e-7)
        assert p['attention'] is None and 'joint' not in p
    model.set_epoch(119)
    with torch.inference_mode():
        for code in range(1,16):
            mask=mask_from_code(code)[None]
            p=model(x,True,modality_mask=mask)
            assert p['output'].shape==(1,3,*small.patch_size) and torch.isfinite(p['output']).all()
            assert torch.all(p['weights'][:,:,:4].masked_select(~mask[:,None].expand(-1,3,-1))==0)
            torch.testing.assert_close(p['weights'].sum(-1),torch.ones(1,3))
            assert p['expert_mask'][0,4] and torch.all(p['beta']==1)
            assert torch.all(p['attention'].diagonal(dim1=1,dim2=3)==0)
            torch.testing.assert_close(p['final'],model.semantic.fuse(p['experts'],p['weights']),rtol=0,atol=0)
            stats=output_summary(p)
            assert stats['joint_logit_rms'] is None and stats['joint_present'] is False
            routes=aggregate_case_routes([stats],'synthetic',code)
            assert len(routes)==15 and all(r['beta']==1 and r['joint_weight']==0 for r in routes)
            assert all(r['effective_weight']==r['weight'] for r in routes)
            dirty=x.clone();dirty[:,~mask[0]]=1234
            torch.testing.assert_close(p['final'],model(dirty,modality_mask=mask),rtol=0,atol=0)
    report['all15_masks_forward_routing_and_hidden_channel_invariance']=True
    captured=[]
    handle=model.semantic.experts[4].register_forward_pre_hook(lambda m,a:captured.append(a[0].detach().clone()))
    mask=mask_from_code(2)[None]
    with torch.inference_mode():model(x,modality_mask=mask)
    handle.remove()
    assert captured[0].shape[1]==8
    expected=mask[:,:,None,None,None].expand_as(x).to(x)
    torch.testing.assert_close(captured[0][:,4:],expected,rtol=0,atol=0)
    assert torch.count_nonzero(captured[0][:,:4][:,~mask[0]])==0
    report['G_four_MRI_plus_four_availability_maps']=True
    # Nonzero test correction heads expose attention gradients beyond the
    # intentionally identity-initialized first communication step.
    with torch.no_grad():
        for h in model.communication.correct:torch.nn.init.normal_(h[-1].weight,std=.02)
    model.train();model.zero_grad(set_to_none=True)
    mask=torch.stack((mask_from_code(1),mask_from_code(8)))
    p=model(torch.cat((x,x*1.2)),True,modality_mask=mask)
    p['final'].square().mean().backward()
    assert all(torch.isfinite(q.grad).all() for q in model.parameters() if q.grad is not None)
    assert model.semantic.experts[1].encoder_stages[0][0].weight.grad is None
    assert model.semantic.experts[2].encoder_stages[0][0].weight.grad is None
    for module in [model.semantic.experts[0],model.semantic.experts[3],model.semantic.experts[4],
                   model.communication.value]:
        assert any(q.grad is not None and torch.count_nonzero(q.grad)>0 for q in module.parameters())
    stats=experts_only_summary(model,p)
    assert not any('joint' in k for k in stats['gradients'])
    # One physical expert + G has exactly one sender per receiver, so its
    # peer-softmax is identically 1 and Q/K gradients correctly vanish. Use
    # full availability to test learned selection among multiple senders.
    model.zero_grad(set_to_none=True)
    full=model(x,True)
    full['final'].square().mean().backward()
    for module in (model.communication.query,model.communication.key):
        assert any(q.grad is not None and torch.count_nonzero(q.grad)>0 for q in module.parameters())
    report['heterogeneous_mask_backward_and_live_attention_gradients']=True
    try:model(x,modality_mask=torch.zeros(1,4,dtype=torch.bool))
    except ValueError:pass
    else:raise AssertionError('Empty physical input was accepted')
    del model,p,full;gc.collect()


def training_and_export_checks(plans,data,report,work):
    p=deepcopy(plans);p['configurations']['3d_fullres']['patch_size']=[32,64,16]
    p['configurations']['3d_fullres']['batch_size']=2
    t=make(p,data)
    assert Path(t.output_folder).is_relative_to(work/'results')
    assert t.num_epochs==300 and not t.enable_deep_supervision and t._get_deep_supervision_scales() is None
    assert t.loss.__class__.__name__=='ExclusiveHierarchyLoss'
    assert t.train_step.__func__.__module__.endswith('agent_v1.trainer')
    t.set_deep_supervision_enabled(False)
    batch=batch_for(t)
    t.current_epoch=0;t._agent_steps=0;t.on_train_epoch_start()
    warm=t.train_step(batch);assert np.isfinite(warm['loss'])
    assert warm['fusion']==0 and warm['refined_aux_coefficient']==0
    # Verify the inherited local/fusion loss budgets without a J target.
    t.network.eval();t.network.set_epoch(50)
    with torch.no_grad():
        parts=t.network(batch['data'],True)
        loss,values=t.compute_agent_loss(parts,batch['target'])
        main=t.loss(parts['final'],batch['target'])
        fusion=t.region_aux_loss(parts['final'],batch['target'])
        expected=main+.25*values['expert_mean']+.25*fusion
        torch.testing.assert_close(loss,expected)
    t.current_epoch=99;t._agent_steps=99;t.on_train_epoch_start();t.train_step(batch)
    t.inference_allowed_mirroring_axes=(0,)
    checkpoint=Path(t.output_folder)/'checkpoint_boundary.pth';t.save_checkpoint(checkpoint)
    resumed=make(p,data,resume=True);resumed.load_checkpoint(checkpoint)
    assert resumed.current_epoch==100 and resumed._agent_steps==100
    assert state_digest(resumed.network)==state_digest(t.network)
    assert int(resumed.network.stage_epoch)==99 and not bool(resumed.network.communication_active)
    t.current_epoch=100
    for trainer in (t,resumed):trainer.on_train_epoch_start();trainer.train_step(batch)
    assert state_digest(resumed.network)==state_digest(t.network)
    report['epoch99_to100_resume_and_next_optimizer_step_identical']=True
    del resumed;gc.collect()
    t.current_epoch=119;t._agent_steps=119;t.on_train_epoch_start()
    result=t.train_step(batch);assert np.isfinite(result['loss'])
    assert result['refined_aux_coefficient']==.1
    t._agent_probe_every=25
    validation=t.validation_step(batch)
    assert len(validation['tp_hard'])==3 and np.isfinite(validation['loss'])
    t.on_train_epoch_end([result])
    # Exercise custom validation probes; the standard aggregation receives an
    # actual synthetic validation_step output, not an empty mocked result.
    t._agent_probe_every=25
    t.current_epoch=124
    t.on_validation_epoch_end([validation])
    t.current_epoch=119;t._agent_steps=120;t.network.set_epoch(119)
    checkpoint=Path(t.output_folder)/'checkpoint_best.pth';t.save_checkpoint(checkpoint)
    base=Path(t.output_folder_base)
    (base/'plans.json').write_text(json.dumps(t.plans_manager.plans))
    (base/'dataset.json').write_text(json.dumps(data))
    (Path(t.output_folder)/'agent_experiment.json').write_text(json.dumps(t.training_contract(),indent=2))
    report['trainer_noDS_loss_phase_backward_validation_and_epoch_hooks']=True
    report['actual_training_contract']=t.training_contract()
    del t,batch,parts;gc.collect()
    predictor=load_predictor(base,'checkpoint_best.pth',device=torch.device('cpu'),mirroring=True)
    assert int(predictor.network.architecture_version)==7
    assert not hasattr(predictor.network,'decoder')
    shape=(24,28,12);affine=np.diag([-1.,1.,1.,1.]);affine[:3,3]=[23,-17,8]
    raw=np.zeros(shape,np.float32);raw[2:-2,3:-3,2:-2]=np.random.default_rng(9).uniform(1,20,(20,22,8))
    path=work/'only_T1ce.nii.gz';nib.save(nib.Nifti1Image(raw,affine),path)
    prefix=work/'predictions'/'synthetic_T1ce'
    routes,props=predictor.predict_available_case({'case_id':'synthetic_T1ce','modalities':{'T1ce':str(path)}},prefix)
    image=nib.load(str(prefix)+'.nii.gz')
    assert image.shape==shape and np.allclose(image.affine,affine,rtol=0,atol=1e-4)
    assert set(np.unique(np.asanyarray(image.dataobj))).issubset({0,1,2,3})
    assert int(predictor.network.stage_epoch)==119 and bool(predictor.network.communication_active)
    assert predictor._case_mask is None
    rows=aggregate_case_routes(routes,'synthetic_T1ce',2)
    assert len(rows)==15 and all(r['joint_weight']==0 for r in rows)
    report['checkpoint_discovery_strict_load_TTA_single_available_file_geometry_and_routes']=True


def full_patch_smoke(plans,data,report):
    t=make(plans,data,device=torch.device('cuda'))
    t.network.set_epoch(119)
    batch=batch_for(t,t.batch_size)
    t.optimizer.zero_grad(set_to_none=True)
    with torch.autocast('cuda'):
        parts=t.network(batch['data'],True,modality_mask=torch.ones(t.batch_size,4,dtype=torch.bool,device=t.device))
        loss,components=t.compute_agent_loss(parts,batch['target'])
    t.grad_scaler.scale(loss).backward();t.grad_scaler.unscale_(t.optimizer)
    assert torch.isfinite(loss) and all(torch.isfinite(q.grad).all() for q in t.network.parameters() if q.grad is not None)
    torch.nn.utils.clip_grad_norm_(t.network.parameters(),12)
    t.grad_scaler.step(t.optimizer);t.grad_scaler.update();torch.cuda.synchronize()
    report['full_plan_cuda_synthetic_step']=dict(patch=list(t.configuration_manager.patch_size),
        batch_size=t.batch_size,all_five_experts_active=True,communication_active=True,
        loss=float(loss.detach()),peak_allocated_GiB=torch.cuda.max_memory_allocated()/1024**3,
        peak_reserved_GiB=torch.cuda.max_memory_reserved()/1024**3,device=torch.cuda.get_device_name(0))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-root',type=Path,default=Path('/data02/chujingyi/data/nnUNet_preprocessed/Dataset137_BraTS2021'))
    parser.add_argument('--report',type=Path,required=True)
    parser.add_argument('--full-patch-smoke',action='store_true')
    args=parser.parse_args();torch.set_num_threads(2)
    report={'status':'running','patient_data_used':False,'full_training_started':False,'torch':torch.__version__}
    try:
        plans=json.loads((args.dataset_root/'nnUNetPlans.json').read_text())
        data=json.loads((args.dataset_root/'dataset.json').read_text())
        with tempfile.TemporaryDirectory(prefix='brats_five_experts_check_') as tmp:
            work=Path(tmp);os.environ['nnUNet_results']=str(work/'results')
            os.environ['nnUNet_wandb_enabled']='false'
            os.environ['BRATS_DIAG_EVERY']='1';os.environ['BRATS_ABLATE_EVERY']='0'
            if args.full_patch_smoke:
                if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
                full_patch_smoke(plans,data,report)
            else:
                assert recursive_find_python_class(str(Path(nnunetv2.__path__[0])/'training/nnUNetTrainer'),Trainer.__name__,'nnunetv2.training.nnUNetTrainer') is Trainer
                architecture_checks(plans,report)
                training_and_export_checks(plans,data,report,work)
        report['status']='passed'
    except BaseException:
        report['status']='failed';report['error']=traceback.format_exc();raise
    finally:
        args.report.parent.mkdir(parents=True,exist_ok=True)
        args.report.write_text(json.dumps(report,indent=2)+'\n')
        print(report['status']+': '+str(args.report),flush=True)


if __name__=='__main__':main()
