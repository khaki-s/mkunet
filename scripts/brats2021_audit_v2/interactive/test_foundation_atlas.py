"""CPU synthetic integration: real upstream adapter, fake frozen dense refiner."""
import importlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import nibabel as nib
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path[:] = [p for p in sys.path if Path(p or '.').resolve() != ROOT]
folder = ROOT/'nnunetv2/utilities/brats2021_audit_v2/interactive'
name = 'brats_atlas_integration_test'
spec = importlib.util.spec_from_file_location(name, folder/'__init__.py', submodule_search_locations=[str(folder)])
package = importlib.util.module_from_spec(spec);sys.modules[name]=package;spec.loader.exec_module(package)
pipeline = importlib.import_module(name+'.foundation_prompt_v2.atlas_pipeline')


class FakeRefiner:
    calls = 0
    releases = 0
    def __init__(self, model_folder, device):
        pass
    def predict_head(self, mri, current, head):
        type(self).calls += 1
        # The synthetic image encodes tissue identity; no GT argument/file read.
        anatomy = np.rint(mri[0]).astype(np.uint8)
        return pipeline.adapter._region(anatomy,head).astype(np.uint8), 1
    def close(self):
        type(self).releases += 1


class AtlasIntegration(unittest.TestCase):
    def test_wt_primary_criterion_allows_secondary_region_tradeoffs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            artifact=root/'artifact.json';pipeline.write_json(artifact,{'synthetic':True})
            gains={'WT':.1,'TC':-.02,'ET':-.01,'ED':-.2,'NCR':-.3}
            result={'schema':'atlas_dense_score_v1','split':'validation','full_cohort':True,'case_count':80,
                    'predictions':pipeline.file_identity(artifact),'truth_manifest':pipeline.file_identity(artifact),
                    'checkpoint':pipeline.file_identity(artifact),'contract':{'synthetic':True},
                    'regions':{h:{'mean_gain_pp':g,'secondary_empty_as_one_mean_gain_pp':g,
                                  'empty_gt_new_fp_voxels':0} for h,g in gains.items()}}
            score=root/'score.json';pipeline.write_json(score,result)
            args=SimpleNamespace(score=str(score),output=str(root/'wt_lock.json'),criterion='wt-primary')
            path=pipeline.lock(args)
            self.assertTrue(pipeline.read_json(path)['passed'])
            self.assertFalse(pipeline.read_json(path)['all_three_improved'])
            args.criterion='all-three';args.output=str(root/'three_lock.json')
            with self.assertRaisesRegex(ValueError,'all-three'):
                pipeline.lock(args)
            self.assertFalse(Path(args.output).exists())

    def test_empty_regions_and_mri_binding(self):
        with tempfile.TemporaryDirectory(prefix='atlas-score-empty-') as temporary:
            root=Path(temporary)
            splits=root/'splits.json';splits.write_text(json.dumps([{'train':['train_case'],'val':['val_case']}]))
            checkpoint=root/'checkpoint';checkpoint.write_text('scoring-only dummy')
            gt=np.zeros((8,8,8),np.uint8);gt[1:7,1:7,1:7]=1;gt[2:6,2:6,2:6]=2
            after=gt.copy();after[3:5,3:5,3:5]=3
            def save(path,array):
                nib.save(nib.Nifti1Image(array,np.eye(4)),path)
                return pipeline.file_identity(path)
            row={'case_id':'val_case','availability':[1]*4,'prediction':save(root/'initial.nii.gz',gt),
                 'gt':save(root/'truth.nii.gz',gt),'modalities':{}}
            for i,modality in enumerate(('T1','T1ce','T2','FLAIR')):
                row['modalities'][modality]=save(root/f'mri{i}.nii.gz',np.full(gt.shape,i,dtype=np.float32))
            meta={'schema':'foundation_cases_v2','split':'validation','fold':0,'contains_gt':True,
                  'splits_file':pipeline.file_identity(splits),'initial_checkpoint':pipeline.file_identity(checkpoint),'cases':[row]}
            full=root/'full.json';deploy=root/'deploy.json'
            pipeline.write_json(full,meta)
            pipeline.write_json(deploy,{**meta,'contains_gt':False,'cases':[{k:v for k,v in row.items() if k!='gt'}]})
            final=save(root/'final.nii.gz',after)
            index=root/'predictions.json'
            pipeline.write_json(index,{'binding':{'schema':'atlas_dense_predictions_v1',
                                 'source':pipeline.file_identity(deploy),'checkpoint':pipeline.file_identity(checkpoint),'contract':{}},
                                 'completed':{'val_case':{'files':[row['prediction'],final],
                                 'baseline':row['prediction'],'final':final,'total_seconds_including_export':0,'nninteractive_calls':0}}})
            path=pipeline.score(SimpleNamespace(predictions=str(index),manifest=str(full),output=str(root/'score')))
            result=pipeline.read_json(path);et=result['regions']['ET']
            self.assertIsNone(et['baseline_pct']);self.assertIsNone(et['mean_gain_pp']);self.assertEqual(et['n'],0)
            self.assertEqual(et['secondary_empty_as_one_mean_gain_pp'],-100.)
            self.assertEqual(et['empty_gt_new_fp_voxels'],8)
            self.assertEqual(et['empty_gt_new_fp_case_count'],1)
            raw=next(r for r in result['per_case'] if r['region']=='ET')
            self.assertTrue(raw['baseline_both_empty']);self.assertFalse(raw['final_both_empty'])
            with self.assertRaises(ValueError):
                pipeline.lock(SimpleNamespace(score=str(path),output=str(root/'badlock.json')),expected_cases=1)
            meta['cases'][0]['modalities']['T1']=meta['cases'][0]['modalities']['T1ce']
            bad=root/'bad_mri_source.json';pipeline.write_json(bad,meta)
            with self.assertRaises(ValueError):
                pipeline.score(SimpleNamespace(predictions=str(index),manifest=str(bad),output=str(root/'bad_score')))

    def test_cohort_flow_and_guards(self):
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory(prefix='atlas-cohort-cpu-') as temporary:
            root=Path(temporary)
            model=root/'model';(model/'fold_0').mkdir(parents=True)
            for path in (model/'fold_0/checkpoint_final.pth',model/'plans.json',model/'dataset.json'):
                path.write_text('synthetic weight provenance only; fake model')
            initial=root/'initial.pth';initial.write_text('synthetic base checkpoint')
            splits=root/'splits.json';splits.write_text(json.dumps([{'train':['train_case'],'val':['val_case']}]))
            gt=np.zeros((12,12,12),np.uint8);gt[1:11,1:11,1:11]=1;gt[3:10,3:10,3:10]=2;gt[5:9,5:9,5:9]=3
            baseline=gt.copy();baseline[5:9,5:9,5:7]=2;baseline[0,0,0]=1
            manifests={}
            for split,key in [('train','train_case'),('validation','val_case')]:
                directory=root/key;directory.mkdir()
                def save(path,array):
                    nib.save(nib.Nifti1Image(array,np.eye(4)),path)
                    return pipeline.file_identity(path)
                row={'case_id':key,'availability':[1]*4,
                     'prediction':save(directory/'initial.nii.gz',baseline),
                     'gt':save(directory/'truth.nii.gz',gt), 'modalities':{}}
                for i,modality in enumerate(('T1','T1ce','T2','FLAIR')):
                    image=gt.astype(np.float32) if i==0 else np.random.default_rng(i).normal(size=gt.shape).astype(np.float32)
                    row['modalities'][modality]=save(directory/f'mri{i}.nii.gz',image)
                meta={'schema':'foundation_cases_v2','split':split,'fold':0,
                      'splits_file':pipeline.file_identity(splits),'initial_checkpoint':pipeline.file_identity(initial),
                      'contains_gt':True,'cases':[row]}
                full=root/f'{split}.json';pipeline.write_json(full,meta)
                clean=root/f'{split}.deployment.json'
                pipeline.write_json(clean,{**meta,'contains_gt':False,'cases':[{k:v for k,v in row.items() if k!='gt'}]})
                manifests[split]=(full,clean)
            args=SimpleNamespace(manifest=str(manifests['train'][0]),output=str(root/'pairs'),
                                 model_folder=str(model),device='cpu',limit_cases=None)
            pairs=pipeline.collect(args,refiner_factory=FakeRefiner)
            calls=FakeRefiner.calls
            pipeline.collect(args,refiner_factory=FakeRefiner)
            self.assertEqual(calls,FakeRefiner.calls,'Completed collection should resume without new model calls')
            Path(pairs).unlink()
            pipeline.collect(args,refiner_factory=FakeRefiner)
            self.assertEqual(len(pipeline.read_json(pairs)['cases']),3)
            self.assertEqual(calls,FakeRefiner.calls,'Resume must repair pair index without new inference')
            checkpoint=pipeline.train(SimpleNamespace(pairs=str(pairs),output=str(root/'adapter'),
                                      epochs=3,steps_per_epoch=4,patch_size=(12,12,12),base_ch=4,
                                      model_folder=str(model),device='cpu'))
            # Separate GPU jobs must optimize the same independent heads as a
            # serial run, including per-head sampling and initialization seeds.
            head_paths=[]
            for head in pipeline.HEADS:
                head_paths.append(pipeline.train(SimpleNamespace(pairs=str(pairs),
                    output=str(root/f'adapter_{head}'), heads=[head], epochs=3,
                    steps_per_epoch=4,patch_size=(12,12,12),base_ch=4,
                    model_folder=str(model),device='cpu',cache_pairs=1)))
            merged=pipeline.adapter.merge_adapters(head_paths,root/'adapter_merged')
            serial_state=torch.load(checkpoint,weights_only=True)
            merged_state=torch.load(merged,weights_only=True)
            for h,parameters in serial_state['models'].items():
                for name,value in parameters.items():
                    torch.testing.assert_close(value,merged_state['models'][h][name],rtol=0,atol=0)
            checkpoint=merged
            p_args=SimpleNamespace(manifest=str(manifests['validation'][1]),output=str(root/'prediction'),
                                   checkpoint=str(checkpoint),model_folder=str(model),device='cpu',
                                   limit_cases=None,validation_lock=None)
            prediction=pipeline.predict(p_args,refiner_factory=FakeRefiner)
            calls=FakeRefiner.calls
            pipeline.predict(p_args,refiner_factory=FakeRefiner)
            self.assertEqual(calls,FakeRefiner.calls,'Completed prediction should not rerun')
            score=pipeline.score(SimpleNamespace(predictions=str(prediction),manifest=str(manifests['validation'][0]),output=str(root/'score')))
            result=pipeline.read_json(score)
            self.assertEqual(result['case_count'],1)
            self.assertGreater(result['regions']['ET']['mean_gain_pp'],0)
            lock=pipeline.lock(SimpleNamespace(score=str(score),output=str(root/'fixture_lock.json')),expected_cases=1)
            self.assertTrue(pipeline.read_json(lock)['passed'])
            # The production lock is still strictly 80 independent validation cases.
            with self.assertRaises(ValueError):
                pipeline.lock(SimpleNamespace(score=str(score),output=str(root/'not_val80.json')))
            with self.assertRaises(ValueError):
                pipeline._check_lock(None,result['contract'],checkpoint)
            bad=SimpleNamespace(**vars(p_args));bad.manifest=str(manifests['validation'][0]);bad.output=str(root/'bad')
            with self.assertRaises(ValueError):
                pipeline.predict(bad,refiner_factory=FakeRefiner)
            wrong=SimpleNamespace(**vars(args));wrong.manifest=str(manifests['validation'][0]);wrong.output=str(root/'bad_collect')
            with self.assertRaises(ValueError):
                pipeline.collect(wrong,refiner_factory=FakeRefiner)
            # Saved artefact mutation must invalidate resume.
            first=pipeline.read_json(prediction)['completed']['val_case']['final']['path']
            with open(first,'ab') as f:f.write(b'tamper')
            with self.assertRaises(ValueError):
                pipeline.predict(p_args,refiner_factory=FakeRefiner)
            self.assertEqual(FakeRefiner.releases,2)
            check={'status':'passed','synthetic_only':True,'actual_patient_training':False,'gpu_used':False,
                   'actual_upstream_adapter_optimized':True,'fake_foundation_model':True,
                   'checks':['collect/train/predict/score/lock complete integration',
                             'hash resume without duplicate calls','GT deployment input rejected',
                             'validation cannot become training','incomplete val80 lock rejected',
                             'test without lock rejected','tampered saved output rejected',
                             'per-case engine released'],
                   'synthetic_result_not_performance_evidence':result['regions']}
            output=Path('/data02/chujingyi/brats_dataset_audit/foundation_autoprompt_20261113/verification/atlas_cohort_cpu.json')
            pipeline.write_json(output,check)


if __name__=='__main__':
    unittest.main()
