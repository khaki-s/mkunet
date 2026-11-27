"""CPU regressions for memory policy, GPU shards, and independent head merging."""
import importlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import sys

import torch
import foundation_prompt

common = importlib.import_module(foundation_prompt.name + '.foundation_prompt_v2.common')
parallel = importlib.import_module(foundation_prompt.name + '.foundation_prompt_v2.parallel')
runtime = importlib.import_module(foundation_prompt.name + '.inference_runtime')
adapter = importlib.import_module(foundation_prompt.name + '.foundation_prompt_v2.atlas_adapter')


class RuntimeTests(unittest.TestCase):
    def test_loader_warmup_uses_configured_patch_and_memory_policy(self):
        calls = []
        class FakeSession:
            def __init__(self, **kwargs):
                self.configuration_manager = SimpleNamespace(
                    configuration={'patch_size': [192]*3}, pool_op_kernel_sizes=[[1]*3, [2]*3])
                self.network = torch.nn.Linear(1, 1)
            def warmup(self):
                calls.append((list(self.configuration_manager.configuration['patch_size']),
                              torch.backends.cudnn.benchmark, self.network.training))
            def initialize_from_trained_model_folder(self, folder):
                torch.backends.cudnn.benchmark = True
                self.warmup()
        name = 'nnInteractive.inference.inference_session'
        previous = torch.backends.cudnn.benchmark
        try:
            with patch.dict(sys.modules, {name: SimpleNamespace(nnInteractiveInferenceSession=FakeSession)}):
                engine = runtime.new_session('/synthetic', 'cpu')
            self.assertEqual(calls, [([192]*3, False, False)])
            self.assertNotIn('warmup', engine.__dict__)
        finally:
            torch.backends.cudnn.benchmark = previous

    def test_memory_policy_and_exception_restore(self):
        manager = SimpleNamespace(configuration={'patch_size': [192]*3},
                                  pool_op_kernel_sizes=[[1]*3, [2]*3, [2]*3, [2]*3, [2]*3, [2]*3])
        engine = SimpleNamespace(configuration_manager=manager, network=torch.nn.Linear(1, 1))
        runtime.configure_session(engine)
        self.assertEqual(manager.configuration['patch_size'], [192]*3)
        self.assertFalse(engine.network.training)
        self.assertFalse(next(engine.network.parameters()).requires_grad)
        manager.configuration['patch_size'] = [96]*3
        with self.assertRaisesRegex(ValueError, 'checkpoint-native 192'):
            runtime.configure_session(engine)
        manager.configuration['patch_size'] = [192]*3
        previous = torch.backends.cudnn.benchmark
        torch.backends.cudnn.benchmark = True
        try:
            with self.assertRaisesRegex(torch.OutOfMemoryError, 'do not pool memory'):
                with runtime.prediction_context(engine):
                    self.assertFalse(torch.backends.cudnn.benchmark)
                    raise torch.OutOfMemoryError('synthetic OOM')
            self.assertTrue(torch.backends.cudnn.benchmark)
        finally:
            torch.backends.cudnn.benchmark = previous

    def make_shards(self, root, dense=False):
        splits = root / 'splits.json'
        common.write_json(splits, [{'train': ['a', 'b', 'c', 'd'], 'val': ['v']}])
        source = root / 'train.json'
        rows = [{'case_id': k} for k in 'abcd']
        common.write_json(source, {'schema': 'foundation_cases_v2', 'split': 'train', 'fold': 0,
                                  'splits_file': common.file_identity(splits), 'cases': rows})
        paths = []
        for i in range(2):
            binding = {'schema': 'atlas_dense_collect_v1' if dense else 'foundation_feedback_v2',
                       'source' if dense else 'source_manifest': common.file_identity(source),
                       'contract': {'synthetic': True}, 'limit_cases': None}
            if not dense:
                binding['save_adapter_pairs'] = False
            selected = parallel.partition(rows, SimpleNamespace(num_shards=2, shard_index=i), binding)
            completed = {}
            for row in selected:
                artifact = root / (row['case_id'] + '.txt')
                artifact.write_text('fixture')
                completed[row['case_id']] = {'files': [common.file_identity(artifact)],
                                             'pairs' if dense else 'adapter_rows': []}
            path = root / f'shard{i}.json'
            common.write_json(path, {'binding': binding, 'completed': completed})
            paths.append(path)
        return paths

    def test_disjoint_complete_shards_and_tamper(self):
        for dense in (False, True):
            with self.subTest(dense=dense), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                paths = self.make_shards(root, dense)
                result = parallel.merge_collections(paths, root / 'merged', dense=dense)
                data = common.read_json(result)
                if dense:
                    data = common.read_json(root / 'merged/collection.json')
                self.assertTrue(data['full_cohort'])
                self.assertEqual(set(data['completed']), set('abcd'))
                with self.assertRaisesRegex(ValueError, 'repeated shard'):
                    parallel.merge_collections([paths[0], paths[0]], root / 'duplicate', dense=dense)
                broken = common.read_json(paths[1]); del broken['completed']['b']
                common.write_json(paths[1], broken)
                with self.assertRaisesRegex(ValueError, 'incomplete'):
                    parallel.merge_collections(paths, root / 'partial', dense=dense)
                # Restore the complete index, then corrupt an actual artifact.
                self.make_shards(root, dense)
                (root / 'a.txt').write_text('tampered')
                with self.assertRaises(ValueError):
                    parallel.merge_collections(paths, root / 'corrupt', dense=dense)

    def test_adapter_merge_requires_three_compatible_heads(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = []
            for head in range(3):
                path = root / f'head{head}.pth'
                torch.save({'metadata': {'schema': 'atlas_bratspostfusion_v1', 'heads': [str(head)],
                                        'cohort_contract': {'patch': [192]*3}, 'sampling': 'disagreement'},
                            'models': {str(head): {'weight': torch.tensor([head])}}}, path)
                paths.append(path)
            merged = adapter.merge_adapters(paths, root / 'merged')
            state = torch.load(merged, weights_only=True)
            self.assertEqual(set(state['models']), {'0', '1', '2'})
            with self.assertRaisesRegex(ValueError, 'exactly WT'):
                adapter.merge_adapters(paths[:2], root / 'incomplete')
            state = torch.load(paths[2], weights_only=True)
            state['metadata']['sampling'] = 'legacy'; torch.save(state, paths[2])
            with self.assertRaisesRegex(ValueError, 'different training'):
                adapter.merge_adapters(paths, root / 'mismatch')


if __name__ == '__main__':
    unittest.main()
