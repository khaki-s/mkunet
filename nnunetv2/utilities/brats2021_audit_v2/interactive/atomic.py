"""Explicit ED/NCR point replacement across two existing binary-region tools."""
from contextlib import contextmanager

import torch

from .schemas import ATOMIC_REGIONS


@contextmanager
def atomic_point_transaction(session, voxel, class_name):
    """Allow a user to replace old point-class constraints, with undo and rollback.

    Raw point IDs remain immutable historical annotations. Only confirmed_mask
    defines the prompts replayed to the backend; old bits at this exact voxel
    are superseded. The first undo snapshot retains them for full restoration.
    """
    mapping = {'ED': (('WT', 'include'), ('TC', 'exclude')),
               'NCR': (('TC', 'include'), ('ET', 'exclude'))}
    if class_name not in mapping:
        raise ValueError('Explicit ED/NCR assignment required')
    if (not isinstance(voxel, (tuple, list)) or len(voxel) != 3
            or any(type(v) is not int or not 0 <= v < session.shape[a] for a, v in enumerate(voxel))):
        raise ValueError('Invalid atomic assignment point')
    voxel = tuple(voxel)
    index = (slice(None), *voxel)
    previous_labels = session.confirmed_labels[index].clone()
    previous_known = session.confirmed_mask[index].clone()
    count = len(session.history)
    used_ids = {pid for record in session.history for key in ('positive_point_ids', 'negative_point_ids')
                for pid in record['action'][key]}
    superseded = [p.point_id for p in session.points.values() if p.voxel == voxel and p.point_id in used_ids]
    session.confirmed_labels[index] = 0
    session.confirmed_mask[index] = False
    try:
        yield
        records = session.history[count:]
        if len(records) != 2:
            raise ValueError('Atomic assignment requires exactly two region tool calls')
        for record, (target, operation) in zip(records, mapping[class_name]):
            action = record['action']
            ids = list(action['positive_point_ids']) + list(action['negative_point_ids'])
            if (action['target'] != target or action['operation'] != operation
                    or len(ids) != 1 or session.points[ids[0]].voxel != voxel
                    or action.get('box_prompt')):
                raise ValueError('Region actions do not implement the explicit atomic assignment')
        # Undoing the two tools must restore the OLD constraints, not the
        # temporarily cleared state used for this new interpretation.
        first = session._undo_stack[count]
        local = tuple(v - a for v, a in zip(voxel, first['roi'].start))
        first['labels'][(slice(None), *local)] = previous_labels
        first['known'][(slice(None), *local)] = previous_known
        desired = torch.tensor(ATOMIC_REGIONS[class_name], dtype=session.confirmed_labels.dtype)
        if not torch.equal((session.logits[index] > 0), desired.bool()):
            raise ValueError('Region tools did not satisfy the explicit atomic point class')
        session.confirmed_labels[index] = desired
        session.confirmed_mask[index] = True
        for record in records:
            record['atomic_assignment'] = {'class_name': class_name, 'voxel': list(voxel),
                'superseded_point_ids': superseded, 'human_prompt_count': 1, 'visual_tool_calls': 2}
    except Exception:
        while len(session.history) > count:
            session.undo()
        session.confirmed_labels[index] = previous_labels
        session.confirmed_mask[index] = previous_known
        raise
