"""CPU correctness tests. Synthetic inputs are not segmentation performance evidence."""
import importlib
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import nibabel as nib
import numpy as np
import torch
import foundation_prompt

dense=importlib.import_module(foundation_prompt.name+'.foundation_prompt_v2.sam_dense')
fusion=importlib.import_module(foundation_prompt.name+'.foundation_prompt_v2.sam_full_fusion')
p=importlib.import_module(foundation_prompt.name+'.foundation_prompt_v2.sam_fusion_pipeline')


class FakeSAM(torch.nn.Module):
    def __init__(self):
        super().__init__();self.weight=torch.nn.Parameter(torch.ones(1));self.calls=[]
        self.inst_interactive_predictor=SimpleNamespace(model=SimpleNamespace(
            sam_prompt_encoder=SimpleNamespace(mask_input_size=(8,8))))
    def predict_inst(self,state,**kwargs):
        self.calls.append(kwargs)
        shape=state['shape'];mask=np.zeros(shape,bool)
        x,y=np.round(kwargs['point_coords'][0]).astype(int);mask[max(0,y-1):y+2,max(0,x-1):x+2]=1
        return np.stack([mask,mask,mask]),np.array([.1,.9,.2]),None


class FakeProcessor:
    resolution=1008
    def __init__(self):self.sizes=[]
    def set_image(self,image):
        self.sizes.append(image.size);return {'shape':(image.height,image.width)}


class NativeSamTests(unittest.TestCase):
    def test_full_slice_geometry_and_no_gt_api(self):
        model,processor=FakeSAM(),FakeProcessor()
        engine=dense.SamDenseRefiner('cpu',model=model,processor=processor)
        current=np.zeros((17,19,5),np.uint8);current[5:12,6:14,2]=3
        mri=np.random.default_rng(5).normal(size=(4,*current.shape)).astype(np.float32)
        outputs,metadata=engine.generate(mri,current)
        self.assertEqual(metadata['sam_image_resolution'],1008)
        self.assertEqual(metadata['native_shape'],[17,19,5])
        self.assertFalse(metadata['gt_in_prompts'])
        self.assertTrue(metadata['frozen_parameters'])
        self.assertEqual(processor.sizes,[(19,17),(19,17)])
        self.assertEqual(metadata['sam_predict_calls'],6)
        for variants in outputs.values():
            for result in variants.values():
                self.assertEqual(result.shape,current.shape)
                self.assertFalse(result[:,:,:2].any());self.assertFalse(result[:,:,3:].any())
        for call in model.calls:
            for x,y in call['point_coords']:
                self.assertTrue(current[int(y),int(x),2]>0)
        self.assertTrue(any(c['mask_input'] is None for c in model.calls))
        self.assertTrue(any(c['mask_input'] is not None for c in model.calls))
        with self.assertRaises(TypeError):engine.generate(mri,current,gt=current)
        processor.resolution=512
        with self.assertRaisesRegex(ValueError,'1008'):
            dense.SamDenseRefiner('cpu',model=model,processor=processor)

    def fixture(self,root):
        splits=root/'splits.json';p.write_json(splits,[{'train':['a'],'val':['v']}])
        source=root/'train.json';p.write_json(source,{'schema':'foundation_cases_v2','split':'train','fold':0,
            'splits_file':p.file_identity(splits),'cases':[{'case_id':'a'}]})
        shape=(9,11,7);gt=np.zeros(shape,np.uint8);gt[2:7,3:9,2:5]=1
        base=gt.copy();base[2:4,3:5,2:4]=0
        mri=np.random.default_rng(7).normal(size=(4,*shape)).astype(np.float32)
        archive=root/'image.npz';np.savez_compressed(archive,mri=mri,baseline=base,gt=gt)
        candidate=root/'sam.nii.gz';nib.save(nib.Nifti1Image(gt,np.eye(4)),candidate)
        row={'case_id':'a','head':0,'image':p.file_identity(archive),'candidate':p.file_identity(candidate)}
        collection=root/'collection.json';p.write_json(collection,{'synthetic':True})
        pairs=root/'pairs.json';p.write_json(pairs,{'schema':'sam_fullvolume_pairs_v1','split':'train',
            'source_manifest':p.file_identity(source),'collection':p.file_identity(collection),
            'sam_contract':{'synthetic':True},'variant':'mask_box_point','cases':[row]})
        return row,pairs,mri,base

    def test_lossless_sdf_cache_native_shape_and_corruption_rebuild(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);row,_,_,_=self.fixture(root)
            original=fusion.features(row)
            cached=fusion.features(row,root/'cache')
            reloaded=fusion.features(row,root/'cache')
            for a,b,c in zip(original,cached,reloaded):
                torch.testing.assert_close(a,b,rtol=0,atol=0);torch.testing.assert_close(a,c,rtol=0,atol=0)
                self.assertEqual(a.shape[-3:],(9,11,7))
            cache_file=next((root/'cache').glob('*.npz'))
            with cache_file.open('ab') as f:f.write(b'corrupt derived-cache identity')
            repaired=fusion.features(row,root/'cache')
            for a,b in zip(original,repaired):torch.testing.assert_close(a,b,rtol=0,atol=0)

    def test_full_volume_training_gradients_and_no_sam_ablation(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);row,pairs,mri,base=self.fixture(root)
            original=fusion._model(0,'cpu').state_dict()
            checkpoint=fusion.train(pairs,root/'model',0,epochs=2,device='cpu',cache_cases=1)
            state=torch.load(checkpoint,map_location='cpu',weights_only=True)
            self.assertFalse(state['metadata']['resized']);self.assertFalse(state['metadata']['cropped'])
            self.assertEqual(state['metadata']['native_training_shapes'],{'a':[9,11,7]})
            self.assertTrue(any(not torch.equal(original[k],v) for k,v in state['state_dict'].items()))
            without_sam=fusion.predict(checkpoint,mri,base,base>0,0,device='cpu')
            np.testing.assert_array_equal(without_sam>=.5,base>0)
            history=p.read_json(root/'model/history.json')
            self.assertTrue(all(v['optimizer_steps']==1 for v in history))

    def test_hierarchy_preserves_wt_and_train_cannot_unlock_test(self):
        probabilities=np.random.default_rng(9).uniform(size=(3,8,9,10)).astype(np.float32)
        result=fusion.project(probabilities)
        np.testing.assert_array_equal(result>0,probabilities[0]>=.5)
        self.assertTrue(np.all((result==3)<=(result>=2)))
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);score=root/'score.json'
            p.write_json(score,{'schema':'sam_fullvolume_score_v1','split':'train','full_cohort':False,'case_count':4})
            with self.assertRaisesRegex(ValueError,'independent val80'):
                p.lock(SimpleNamespace(score=score,output=root/'lock.json'))


if __name__=='__main__':unittest.main()
