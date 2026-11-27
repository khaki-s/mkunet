"""Independent case shards: one process/output directory per GPU."""
from pathlib import Path

from .common import read_json, write_json, check_file, file_identity, manifest


def partition(rows, args, binding):
    count, index = getattr(args, 'num_shards', 1), getattr(args, 'shard_index', 0)
    if count < 1 or not 0 <= index < count:
        raise ValueError('Require num_shards >= 1 and 0 <= shard_index < num_shards')
    if count > 1:
        binding['case_partition'] = {'count': count, 'index': index}
    return rows[index::count]


def add_partition_arguments(parser):
    parser.add_argument('--num-shards', type=int, default=1)
    parser.add_argument('--shard-index', type=int, default=0)


def merge_collections(paths, output, *, dense=False):
    """Verify complete, disjoint shards of one protocol before publishing index."""
    root = Path(output)
    if root.exists():
        raise FileExistsError('Use a new merge output directory')
    completed, parts, reference, count = {}, set(), None, None
    expected_schema = 'atlas_dense_collect_v1' if dense else 'foundation_feedback_v2'
    for path in paths:
        data = read_json(path)
        binding = dict(data['binding'])
        part = binding.pop('case_partition', None)
        if binding.get('schema') != expected_schema or part is None:
            raise ValueError('Expected explicitly sharded collections of the requested kind')
        if reference is None:
            reference, count = binding, part['count']
        if binding != reference or part['count'] != count or part['index'] in parts:
            raise ValueError('Mismatched protocol, shard count, or repeated shard index')
        if count < 2 or not 0 <= part['index'] < count:
            raise ValueError('Invalid shard partition')
        source_id = binding['source' if dense else 'source_manifest']
        check_file(source_id)
        source = manifest(source_id['path'], allowed=('train',))
        rows = source['cases'][:binding['limit_cases']] if binding['limit_cases'] else source['cases']
        expected = {r['case_id'] for r in rows[part['index']::count]}
        if set(data['completed']) != expected:
            raise ValueError('Shard is incomplete or contains cases from another partition')
        if completed.keys() & data['completed'].keys():
            raise ValueError('Overlapping case shards')
        for case in data['completed'].values():
            for identity in case['files']:
                check_file(identity)
        completed.update(data['completed'])
        parts.add(part['index'])
    if reference is None or parts != set(range(count)):
        raise ValueError('All shard indices are required')
    reference['merged_from'] = [file_identity(p) for p in paths]
    root.mkdir(parents=True)
    index = root / ('collection.json' if dense else 'dataset.json')
    write_json(index, {'binding': reference, 'completed': completed,
                      'full_cohort': len(completed) == len(source['cases'])})
    pairs = [r for c in completed.values() for r in c['pairs' if dense else 'adapter_rows']]
    if dense or reference['save_adapter_pairs']:
        meta = {'schema': 'foundation_adapter_pairs_v1', 'split': 'train',
                'source_manifest': source_id['path'], 'cases': pairs}
        if dense:
            meta.update(dense_contract=reference['contract'], collection_identity=file_identity(index))
        write_json(root / 'adapter_pairs.json', meta)
    return root / 'adapter_pairs.json' if dense else index
