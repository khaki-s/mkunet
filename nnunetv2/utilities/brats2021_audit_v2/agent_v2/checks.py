"""Bounded synthetic regression checks; never starts a dataset training run."""
import argparse
import copy
import gc
import json
import os
import tempfile
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from ..agent_v1.blocks import ModalityCommunication3D
from ..agent_v1.network import build_agent_network
from ..agent_v1.losses import positive_region_loss
from ..agent_integration_check import batch_for
from ..missing_modality import MASK_CODES, mask_from_code, sample_masks
from ..reproducibility import seed_experiment, state_digest
from ..evaluation.comparison import compare_runs, load_run, check_protocol
from .blocks import ModalityCommunicationV2
from .network import build_agent_v2_network
from .losses import refined_auxiliary_loss, refined_weight
from .diagnostics import residual_summary, no_peer_intervention


def block_checks(device):
    seed_experiment(19)
    mask = torch.tensor([[1,1,0,0],[0,0,1,0]],device=device,dtype=torch.bool)
    states = [torch.randn(2,8,3,4,3,device=device) for _ in range(4)]
    local = [torch.randn(2,3,6,8,6,device=device) for _ in range(4)]
    target = torch.rand(2,3,6,8,6,device=device)
    outcomes = {}
    for null in (False,True):
        old = ModalityCommunication3D(8,4).to(device)
        block = ModalityCommunicationV2(old,null_peer=null).to(device)
        out, attention = block(local,states,mask)
        for a,b in zip(out,local):
            torch.testing.assert_close(a,b,rtol=0,atol=0)
        assert 'rho' not in dict(block.named_parameters())
        optimizer = torch.optim.SGD(block.parameters(),lr=.5)
        changed = []
        for step in range(4):
            optimizer.zero_grad(set_to_none=True)
            out, attention = block(local,states,mask)
            loss,_ = refined_auxiliary_loss(out,target,mask)
            loss.backward()
            changed.append({n:float(sum((p.grad.abs().sum() for p in getattr(block,n).parameters()
                                            if p.grad is not None),start=torch.zeros((),device=device)))
                            for n in ('query','key','value','correct')})
            optimizer.step()
        assert changed[0]['correct']>0 and changed[-1]['value']>0
        if null:
            assert changed[-1]['query']>0 and changed[-1]['key']>0, changed
        else:
            assert changed[-1]['query']==0 and changed[-1]['key']==0, changed
        out,attention = block(local,states,mask)
        no_peer,_ = block(local,states,mask,sender_mask=torch.zeros_like(mask))
        torch.testing.assert_close(out[2][1],no_peer[2][1],rtol=0,atol=0)
        assert (out[0][0]-no_peer[0][0]).abs().max()>0
        if null:
            block(local,states,mask)
            torch.testing.assert_close(attention.sum(-1)+block.last_null_attention,
                                       mask[:,:,None].expand(-1,-1,3).float())
        # Local-only has identical parameters but no dependency on other states.
        local_block=copy.deepcopy(block);local_block.local_only=True
        a,_=local_block(local,states,mask)
        changed_states=[h.clone() for h in states];changed_states[1].add_(100)
        b,_=local_block(local,changed_states,mask)
        torch.testing.assert_close(a[0],b[0],rtol=0,atol=0)
        assert sum(p.numel() for p in block.parameters())==sum(p.numel() for p in local_block.parameters())
        outcomes['null' if null else 'p1']={'identity_init':True,'optimizer_updates':changed,
                                          'single_agent_peer_zero':True,'local_only_invariance':True}
    # Patient-equal normalization, unaffected by number of present branches.
    constants=[z.detach() for z in local]
    got,_=refined_auxiliary_loss(constants,target,mask)
    expected=(.5*(positive_region_loss(constants[0][:1],target[:1])+positive_region_loss(constants[1][:1],target[:1]))
              +positive_region_loss(constants[2][1:],target[1:]))/2
    torch.testing.assert_close(got,expected)
    assert [refined_weight(e) for e in (49,99,100,119,299)]==[0,0,0,.1,.1]
    outcomes['loss_patient_normalization_and_ramp']=True
    return outcomes


def network_checks(config,device):
    seed_experiment(0)
    old=build_agent_network(config,communication=True).to(device).eval();old.set_epoch(119)
    seed_experiment(0)
    net=build_agent_v2_network(config,communication=True).to(device).eval();net.set_epoch(119)
    shared=[n for n in old.state_dict() if not n.startswith('communication.')]
    assert all(torch.equal(old.state_dict()[n],net.state_dict()[n]) for n in shared)
    x=torch.randn(1,4,*config.patch_size,device=device)
    hits=[0]*4
    def hook(m):
        def count(module,args,result): hits[m]+=args[0].shape[0]
        return count
    handles=[e.register_forward_hook(hook(m)) for m,e in enumerate(net.semantic.experts)]
    with torch.no_grad():
        reference=old(x)[0];torch.testing.assert_close(reference,net(x)[0],rtol=0,atol=0)
        for code in MASK_CODES:
            hits[:]=[0]*4
            mask=mask_from_code(code,device=device)[None]
            parts=net(x,True,modality_mask=mask)
            assert hits==mask.long()[0].tolist()
            assert torch.isfinite(parts['final']).all()
            assert torch.count_nonzero(parts['weights'].masked_select(~mask[:,None,:]))==0
            torch.testing.assert_close(parts['weights'].sum(-1),torch.ones(1,3,device=device))
            hidden=x.masked_fill(~mask[:,:,None,None,None],1e3)
            torch.testing.assert_close(parts['final'],net(hidden,modality_mask=mask)[0],rtol=0,atol=0)
        for h in handles:h.remove()
        try:net(x,modality_mask=torch.zeros(1,4,device=device,dtype=torch.bool))
        except ValueError:pass
        else:raise AssertionError('All missing accepted')
        parts=net(x,True,modality_mask=torch.ones(1,4,device=device,dtype=torch.bool))
        report=residual_summary(parts)
        assert all(v==0 for a in report['agents'].values() for row in a['residual_rms'] for v in row)
        intervention=no_peer_intervention(net,parts,torch.zeros_like(parts['final']))
        assert all(v==0 for row in intervention['peer_logit_rms'] for v in row)
    del old,net;gc.collect()
    return {'shared_initial_weights_identical':True,'initial_full_output_bitwise_v1':True,
            'all15_masking_sparse_execution_hidden_invariance':True,'empty_mask_rejected':True,
            'true_residual_and_no_peer_diagnostics':True}


def trainer_checks(plans,dataset,device,work):
    from nnunetv2.training.nnUNetTrainer.variants.brats2021_audit import nnUNetTrainer_BraTSAgentV2Loss300 as registry
    from nnunetv2.utilities.find_objects import recursive_find_trainer_class_by_name
    classes=[getattr(registry,n) for n in dir(registry) if n.startswith('nnUNetTrainer_')]
    for cls in classes:assert recursive_find_trainer_class_by_name(cls.__name__) is cls
    cls=registry.nnUNetTrainer_BraTSAgentCommV2Loss300
    def make():
        t=cls(copy.deepcopy(plans),'3d_fullres',0,dataset,device=device)
        t.num_iterations_per_epoch=1;t.initialize()
        assert Path(t.output_folder).is_relative_to(work/'results')
        return t
    t=make();batch=batch_for(t)
    param_ids=[id(p) for g in t.optimizer.param_groups for p in g['params']]
    assert len(param_ids)==len(set(param_ids))
    assert set(param_ids)=={id(p) for p in t.network.parameters()}
    assert t.num_epochs==300 and t.lr_scheduler.max_steps==300
    phases=[]
    for epoch in (49,99,119):
        t.current_epoch=epoch;t._agent_steps=epoch;t.on_train_epoch_start()
        result=t.train_step(batch);assert np.isfinite(result['loss'])
        f=work/f'resume_{epoch}.pth';t.save_checkpoint(str(f))
        restored=make();restored.load_checkpoint(f)
        assert restored.current_epoch==epoch+1 and restored._agent_steps==epoch+1
        assert state_digest(t.network)==state_digest(restored.network)
        assert torch.equal(sample_masks(0,t._agent_steps,2),sample_masks(0,restored._agent_steps,2))
        t.current_epoch=epoch+1
        for v in (t,restored):v.on_train_epoch_start();v.train_step(batch)
        assert state_digest(t.network)==state_digest(restored.network),f'Resume diverged at {epoch}'
        phases.append({'saved_epoch':epoch,'next_epoch':epoch+1,'next_step_bitwise_equal':True})
        del restored;gc.collect()
    # New losses/initialization are explicit factors in the strict resume contract.
    altered=make();altered.refined_aux_max=.2
    try:altered.load_checkpoint(f)
    except ValueError:pass
    else:raise AssertionError('Changed loss contract was accepted')
    del altered
    # Confirm every ablation builds, has a finite actual optimizer step and distinct contract.
    factors={}
    for control in classes:
        a=control(copy.deepcopy(plans),'3d_fullres',0,dataset,device=device)
        a.initialize();a.current_epoch=119;a._agent_steps=119*a.num_iterations_per_epoch
        a.on_train_epoch_start();r=a.train_step(batch)
        assert np.isfinite(r['loss'])
        factors[control.__name__]={'local_only':a.local_only,'residual_mode':a.residual_mode,
                                 'null_peer':a.null_peer,'refined_aux_max':a.refined_aux_max,
                                 'trainable_parameters':sum(p.numel() for p in a.network.parameters())}
        del a;gc.collect()
    del t,batch;gc.collect()
    return {'all_parameters_once_in_optimizer':True,'epochs_and_poly_steps':300,'phases':phases,
            'changed_loss_resume_rejected':True,'registered_ablations':factors}


def comparison_checks(work):
    root=work/'compare';root.mkdir()
    ids=['synthetic_a','synthetic_b']
    regions={'WT':[1,2,3],'TC':[2,3],'ET':[3],'ED':[1],'NCR':[2]}
    contract={'case_ids':ids,'mask_codes':list(MASK_CODES),'modalities':['T1','T1ce','T2','FLAIR'],
              'regions':regions,'protocol':'available_files','tile_step':.5,'mirroring':True,'mirror_axes':[0,1,2],
              'gt_sha256':{c:'fixture' for c in ids},'empty_both_dice':None,
              'hd95_empty_policy':'fixture','accumulation':'fixture',
              'provenance':{'split':'test','fold':0,'checkpoint':'checkpoint_best.pth',
                            'checkpoint_sha256':'fixture','inputs':{c:{'T1':'fixture'} for c in ids}}}
    (root/'manifest.json').write_text(json.dumps({'status':'complete','contract':contract}))
    import csv
    with (root/'per_case_metrics.csv').open('w') as f:
        writer=csv.writer(f);writer.writerow(['case_id','mask_code','region','n_modalities','dice','ref_mm3','pred_mm3'])
        for c in ids:
            for m in MASK_CODES:
                for r in regions:writer.writerow([c,m,r,m.bit_count(),.5,10,10])
    _,paired=compare_runs({'a':root,'b':root},ids,work/'comparison_report',reference='a',repetitions=20)
    assert all(r['delta_pp']==r['ci95_low_pp']==r['ci95_high_pp']==0 for r in paired)
    ref=load_run(root,ids);other=copy.deepcopy(ref);other['contract']['mirroring']=False
    try:check_protocol(other,ref)
    except ValueError:pass
    else:raise AssertionError('Mismatched TTA accepted')
    csvfile=root/'per_case_metrics.csv';original=csvfile.read_text()
    csvfile.write_text(original+original.splitlines()[1]+'\n')
    try:load_run(root,ids)
    except ValueError:pass
    else:raise AssertionError('Duplicate row accepted')
    csvfile.write_text('\n'.join(original.splitlines()[:-1])+'\n')
    try:load_run(root,ids)
    except ValueError:pass
    else:raise AssertionError('Missing region row accepted')
    return {'paired_self_zero':True,'tta_mismatch_rejected':True,'duplicate_and_incomplete_rejected':True}


def full_patch_check(plans,dataset,device):
    if device.type!='cuda':raise ValueError('Full patch AMP check requires CUDA')
    from nnunetv2.training.nnUNetTrainer.variants.brats2021_audit.nnUNetTrainer_BraTSAgentV2Loss300 import nnUNetTrainer_BraTSAgentCommV2Loss300
    torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
    t=nnUNetTrainer_BraTSAgentCommV2Loss300(copy.deepcopy(plans),'3d_fullres',0,dataset,device=device)
    t.initialize();t.current_epoch=119;t.network.set_epoch(119)
    batch=batch_for(t,2);started=time.monotonic();steps=[]
    for i in range(2):
        t.optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda'):
            parts=t.network(batch['data'],True,modality_mask=torch.ones(2,4,device=device,dtype=torch.bool))
            loss,components=t.compute_agent_loss(parts,batch['target'])
        before=t.grad_scaler.get_scale();t.grad_scaler.scale(loss).backward();t.grad_scaler.unscale_(t.optimizer)
        assert torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in t.network.parameters() if p.grad is not None)
        torch.nn.utils.clip_grad_norm_(t.network.parameters(),12)
        t.grad_scaler.step(t.optimizer);t.grad_scaler.update()
        assert t.grad_scaler.get_scale()>=before
        steps.append({'loss':float(loss.detach()),'refined_aux':components['refined_aux'],'coefficient':components['refined_aux_coefficient']})
        del parts,loss
    t.network.eval()
    with torch.no_grad(),torch.autocast('cuda'):
        parts=t.network(batch['data'][:1],True,modality_mask=mask_from_code(3,device=device)[None])
        intervention=no_peer_intervention(t.network,parts,batch['target'][0][:1])
    torch.cuda.synchronize()
    report={'patch':list(t.configuration_manager.patch_size),'batch_size':2,'all_four_modalities':True,
            'amp':True,'synthetic_optimizer_steps':steps,'peer_intervention_ran':True,
            'peak_allocated_GiB':torch.cuda.max_memory_allocated()/2**30,
            'peak_reserved_GiB':torch.cuda.max_memory_reserved()/2**30,'seconds':time.monotonic()-started}
    del t,batch,parts,intervention;gc.collect();torch.cuda.empty_cache()
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-root',type=Path,default=Path('/data02/chujingyi/data/nnUNet_preprocessed/Dataset137_BraTS2021'))
    parser.add_argument('--device',choices=('cpu','cuda'),default='cpu')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--full-patch',action='store_true')
    args=parser.parse_args();torch.set_num_threads(2)
    if args.device=='cuda':torch.cuda.set_per_process_memory_fraction(.5)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    report={'status':'running','device':args.device,'torch':torch.__version__,'scope':'bounded synthetic tests only'}
    def save():args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    save()
    plans=json.loads((args.dataset_root/'nnUNetPlans.json').read_text());plans['continue_training']=False
    dataset=json.loads((args.dataset_root/'dataset.json').read_text())
    small=copy.deepcopy(plans);c=small['configurations']['3d_fullres'];c['patch_size']=[64,64,32];c['batch_size']=2
    c['architecture']['arch_kwargs']['features_per_stage']=[8,16,32,64,80,80]
    from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
    config=PlansManager(small).get_configuration('3d_fullres');device=torch.device(args.device)
    try:
        with tempfile.TemporaryDirectory(prefix='brats_v2_tests_') as work:
            # Only change the module-level output root; never point checks at production result folders.
            os.environ['nnUNet_results']=str(Path(work)/'results')
            report['blocks']=block_checks(device);save();print('block checks passed',flush=True)
            report['network']=network_checks(config,device);save();print('network checks passed',flush=True)
            report['trainer']=trainer_checks(small,dataset,device,Path(work));save();print('trainer checks passed',flush=True)
            report['comparison']=comparison_checks(Path(work));save()
            if args.full_patch:report['full_patch']=full_patch_check(plans,dataset,device);save()
        report['status']='passed'
    except BaseException:
        report['status']='failed';report['error']=traceback.format_exc();raise
    finally:save();print(f"{report['status']}: {args.output}",flush=True)


if __name__=='__main__':main()
