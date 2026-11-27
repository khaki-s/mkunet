"""Stage the original-grid regional logits of the three frozen segmenters."""
import argparse
from pathlib import Path

import nibabel as nib
import numpy as np

from ..common import check_file, file_identity, read_json, write_json
from . import probability_features, teacher_fields
from .cohorts import rows

EXTENT = 7.


def case_paths(root, split, row):
    return [probability_features.probability_path(root, name, split, row['case_id'])
            for name in probability_features.EXPERTS]


def convert(probability):
    bound = 1. / (1. + np.exp(EXTENT))
    bounded = np.clip(probability, bound, 1. - bound)
    return np.log(bounded / (1. - bounded)) / EXTENT


def run(args):
    selected = rows(args.cohort, args.split)[args.shard_index::args.num_shards]
    code = file_identity(__file__)
    for row in selected:
        key = row['case_id']
        inputs = case_paths(args.probabilities, args.split, row)
        site = Path(args.output) / args.split / key
        path = site / 'logits.json'
        binding = {'row': row, 'code': code, 'sources': [file_identity(p) for p in inputs]}
        if path.is_file():
            previous = read_json(path)
            if previous['binding'] != binding or previous['gt_used'] is not False:
                raise ValueError('Cached logits came from different expert inputs')
            continue
        check_file(row['prediction'])
        reference = nib.load(row['prediction']['path'])
        logits = np.empty((9, *reference.shape), dtype=np.float16)
        for k, source in enumerate(inputs):
            values, _, _ = probability_features.probability_field(source, row, reference)
            logits[3*k:3*k+3] = convert(values)
        site.mkdir(parents=True, exist_ok=True)
        teacher_fields._write(site / 'logits.npy', logits)
        write_json(path, {'binding': binding, 'shape': list(logits.shape), 'gt_used': False})
        print(f'{key} staged the original grid of all three networks', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ('cohort', 'probabilities', 'output'):
        parser.add_argument('--' + field, required=True)
    parser.add_argument('--split', choices=('train', 'validation'), required=True)
    parser.add_argument('--num-shards', type=int, default=1)
    parser.add_argument('--shard-index', type=int, default=0)
    run(parser.parse_args())
