"""CPU protocol tests. Fakes test integration only, not SAM3 clinical accuracy."""
import importlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import nibabel as nib
import torch

import foundation_prompt

common = importlib.import_module(foundation_prompt.name + '.foundation_prompt_v2.common')
pipeline = importlib.import_module(foundation_prompt.name + '.foundation_prompt_v2.pipeline')
router = importlib.import_module(foundation_prompt.name + '.foundation_prompt_v2.router')
executor = importlib.import_module(foundation_prompt.name + '.foundation_prompt_v2.executor')
workspace = importlib.import_module(foundation_prompt.name + '.foundation_prompt_v2.conv3d_workspace')
SOURCE = '/data02/chujingyi/brats_dataset_audit/foundation_autoprompt_20261113/third_party/PromptMoE/PromptMoE.py'


def action(positive):
    return {'head': 'ET', 'kind': 'point', 'positive': positive,
            'operation': 'include' if positive else 'exclude',
            'voxel': [3, 3, 3] if positive else [2, 2, 2], 'anchor': [3, 3, 3]}


class Proposals:
    def generate(self, mri, current, availability, history, *, affine):
        assert mri.shape[0] == 4 and affine.shape == (4, 4)
        acts = [None]*6 if len(history) >= 2 else [None, None, None, None, action(True), action(False)]
        return {'context': np.arange(8, dtype=np.float32), 'actions': acts, 'metadata': {'fake': True}}


class Corrector:
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, current, prompt, history):
        output = current.copy()
        output[tuple(prompt['voxel'])] = 3 if prompt['positive'] else 2
        return output, {}


class TestFoundationPipeline(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.splits = self.root/'splits.json'
        common.write_json(self.splits, [{'train': ['train_case'], 'val': ['val_case']}])
        self.weights = self.root/'sam3.pt'
        self.weights.write_bytes(b'test-weights-not-a-real-model')
        sam_source = self.root/'sam_source'
        (sam_source/'sam3').mkdir(parents=True)
        (sam_source/'sam3/model_builder.py').write_text('# synthetic protocol fixture')
        fm = self.root/'nni'
        (fm/'fold_0').mkdir(parents=True)
        for name in ('plans.json', 'dataset.json', 'fold_0/checkpoint_final.pth'):
            (fm/name).write_text('{}')
        self.config = self.root/'config.json'
        common.write_json(self.config, {'sam3': {'checkpoint': str(self.weights), 'source_dir': str(sam_source)},
            'correction_protocol': {'model_folder': str(fm)}, 'router_source': SOURCE})
        self.manifests = {}
        for split, key in (('train', 'train_case'), ('validation', 'val_case')):
            folder = self.root/key
            folder.mkdir()
            pred = np.zeros((6, 6, 6), np.uint8)
            pred[1:5, 1:5, 1:5] = 1
            pred[2, 2, 2] = 3
            gt = pred.copy(); gt[3, 3, 3] = 3
            modalities = {}
            for i, name in enumerate(common.MODALITIES):
                path = folder/f'{name}.nii.gz'
                nib.save(nib.Nifti1Image(np.ones(pred.shape, np.float32)*(i+1), np.eye(4)), path)
                modalities[name] = common.file_identity(path)
            for name, labels in (('pred', pred), ('gt', gt)):
                nib.save(nib.Nifti1Image(labels, np.eye(4)), folder/f'{name}.nii.gz')
            row = {'case_id': key, 'modalities': modalities,
                   'prediction': common.file_identity(folder/'pred.nii.gz'),
                   'gt': common.file_identity(folder/'gt.nii.gz'), 'availability': [1]*4}
            value = {'schema': 'foundation_cases_v2', 'split': split, 'fold': 0,
                     'splits_file': common.file_identity(self.splits), 'cases': [row], 'contains_gt': True}
            path = self.root/f'{split}.json'
            common.write_json(path, value)
            clean = self.root/f'{split}.deployment.json'
            common.write_json(clean, {**value, 'contains_gt': False,
                                      'cases': [{k: v for k, v in row.items() if k != 'gt'}]})
            self.manifests[split] = path, clean

    def tearDown(self):
        self.temp.cleanup()

    def train_fixture(self):
        data = pipeline.collect(SimpleNamespace(manifest=str(self.manifests['train'][0]), config=str(self.config),
            output=str(self.root/'data'), rounds=3, pairs=True, save_adapter_pairs=False,
            rollout_policy=None, rollout_margin_pp=0.1, limit_cases=None, device='cpu'),
            proposal_generator=Proposals(), corrector_factory=Corrector)
        checkpoint = pipeline.train(SimpleNamespace(datasets=[str(data)], output=str(self.root/'model'),
             device='cpu', epochs=2, batch_size=2, lr=1e-3, seed=17, threads=1))
        return data, checkpoint

    def test_actual_upstream_class_and_gradients(self):
        model = router.upstream_router(SOURCE, dict(E=6, d_ctx=8, d_hidden=8, n_layers=2, dropout=0.0))
        g = torch.randn(2, 8)
        loss = router.regression_loss(model, g, torch.ones(2, 6), torch.ones(2, 6, dtype=torch.bool),
            torch.ones(2, 6, 6), torch.ones(2, 6, 6, dtype=torch.bool))
        loss.backward()
        self.assertGreater(sum(float(p.grad.abs().sum()) for p in model.parameters() if p.grad is not None), 0)
        u, i, s = model.forward_scores(g)
        torch.testing.assert_close(s[:, 0, 1], u[:, 0]+u[:, 1]+i[:, 0, 1])

    def test_split_and_gt_boundary(self):
        with self.assertRaisesRegex(ValueError, 'WITHOUT GT'):
            common.manifest(self.manifests['validation'][0], deployment=True)
        with self.assertRaisesRegex(ValueError, 'not allowed'):
            common.manifest(self.manifests['validation'][0], allowed=('train',))
        tampered = common.read_json(self.manifests['validation'][0]); tampered['split'] = 'train'
        path = self.root/'forged.json'; common.write_json(path, tampered)
        with self.assertRaisesRegex(ValueError, 'original complete split'):
            common.manifest(path)

    def test_feedback_train_predict_score_and_pair_deferral(self):
        data, checkpoint = self.train_fixture()
        first = next(iter(common.read_json(data)['completed'].values()))['records'][0]
        with np.load(first['state']['path']) as measurements:
            self.assertGreater(measurements['utility'][4], 0)
            self.assertLess(measurements['utility'][5], 0)
        args = SimpleNamespace(manifest=str(self.manifests['validation'][1]), config=str(self.config),
            checkpoint=str(checkpoint), output=str(self.root/'predictions'), margin_pp=0.1,
            pairs=True, limit_cases=None, validation_lock=None, device='cpu')
        # All single actions are rejected; only a two-action plan is positive.
        U = -np.ones(6); S = -np.ones((6, 6)); S[4, 5] = 1
        with patch.object(pipeline, 'router_scores', return_value=(U, S)):
            result = pipeline.predict(args, proposal_generator=Proposals(), corrector_factory=Corrector)
        record = common.read_json(result)['completed']['val_case']
        self.assertEqual(record['budgets']['1']['rounds'], 0)
        self.assertEqual(record['budgets']['3']['rounds'], 2)
        self.assertTrue(record['events'][0]['deferred_for_budget'])
        self.assertTrue(record['stopped'])
        summary = pipeline.score(SimpleNamespace(manifest=str(self.manifests['validation'][0]),
                            predictions=str(result), output=str(self.root/'scores')))
        self.assertEqual(common.read_json(summary)['cases'], 1)
        with self.assertRaisesRegex(ValueError, 'criterion failed'):
            pipeline.validation_lock(SimpleNamespace(summary=str(summary), output=str(self.root/'lock.json')))
        self.assertFalse(common.read_json(self.root/'lock.json')['passed'])

    def test_known_stop_and_unavailable_actions(self):
        actions = [None]*6
        self.assertEqual(router.choose(np.ones(6), np.ones((6, 6)), actions,
                                       remaining=2, margin_pp=0.0)[0], ())
        actions[4] = action(True)
        self.assertEqual(router.choose(-np.ones(6), np.zeros((6, 6)), actions,
                                       remaining=2, margin_pp=0.0)[0], ())

    def test_wt_priority_reward_and_single_action_cap(self):
        config = common.read_json(self.config)
        config.update(reward_profile='wt_priority', max_actions=1)
        common.write_json(self.config, config)
        before = np.zeros((2, 2, 2), np.uint8)
        gt = before.copy()
        gt[0, 0, 0] = 1
        gain, regions = pipeline.candidate_reward(before, gt, gt, pipeline.reward_profile(config))
        self.assertEqual(regions.tolist(), [100., 0., 0., 100., 0.])
        self.assertEqual(gain, 60.)
        _, checkpoint = self.train_fixture()
        args = SimpleNamespace(manifest=self.manifests['validation'][1], config=self.config,
            checkpoint=checkpoint, output=self.root/'single_predictions', margin_pp=0.1,
            pairs=True, limit_cases=None, validation_lock=None, device='cpu')
        with patch.object(pipeline, 'router_scores', return_value=(np.ones(6), np.full((6, 6), 10.))):
            result = pipeline.predict(args, proposal_generator=Proposals(), corrector_factory=Corrector)
        value = common.read_json(result)
        self.assertEqual(value['binding']['deployment']['max_actions'], 1)
        self.assertEqual([v['rounds'] for v in value['completed']['val_case']['budgets'].values()], [0, 1, 1, 1])

    def test_conv3d_tiles_keep_full_grid_and_boundary_values(self):
        rng = torch.Generator().manual_seed(17)
        image = torch.randn((1, 2, 9, 5, 7), generator=rng)
        weight = torch.randn((3, 2, 3, 3, 3), generator=rng)
        bias = torch.randn(3, generator=rng)
        expected = torch.nn.functional.conv3d(image, weight, bias, padding=1)
        for depth in (1, 4, 9, 16):
            actual = workspace.conv3d_depth_tiles(image, weight, bias, depth)
            self.assertEqual(actual.shape, expected.shape)
            torch.testing.assert_close(actual, expected)

    def test_router_scores_exposes_numpy_for_bfloat16(self):
        u = torch.full((1, 6), 0.125, dtype=torch.bfloat16)
        s = torch.full((1, 6, 6), -0.25, dtype=torch.bfloat16)
        model = SimpleNamespace(forward_scores=lambda _: (u, None, s))
        state = {'mean': np.zeros(8), 'std': np.ones(8)}
        single, pair = pipeline.router_scores(model, state, np.ones(8), 'cpu')
        np.testing.assert_array_equal(single, np.full(6, 0.125, np.float32))
        np.testing.assert_array_equal(pair, np.full((6, 6), -0.25, np.float32))

    def test_sharded_collect_empty_partition_and_report_margin(self):
        parallel = importlib.import_module(foundation_prompt.name + '.foundation_prompt_v2.parallel')
        paths = []
        for index in range(2):
            args = SimpleNamespace(manifest=str(self.manifests['train'][0]), config=str(self.config),
                output=str(self.root / f'shard{index}'), rounds=1, pairs=False, save_adapter_pairs=False,
                rollout_policy=None, rollout_margin_pp=0.1, limit_cases=None, device='cpu',
                num_shards=2, shard_index=index, nninteractive_device='cpu')
            paths.append(pipeline.collect(args, proposal_generator=Proposals(), corrector_factory=Corrector))
        merged = parallel.merge_collections(paths, self.root / 'merged')
        self.assertTrue(common.read_json(merged)['full_cohort'])
        report = pipeline.feedback_report(SimpleNamespace(dataset=merged, output=self.root / 'report', margin_pp=100.))
        summary = common.read_json(report)
        self.assertEqual(summary['states_with_beneficial_candidate'], 1)
        self.assertEqual(summary['states_with_candidate_exceeding_margin'], 0)
        self.assertEqual(set(summary['slots']['ET+']['mean_region_delta_pp']), set(common.REGIONS))


if __name__ == '__main__':
    unittest.main()
