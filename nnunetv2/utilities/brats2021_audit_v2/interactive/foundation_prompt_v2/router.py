"""Load PromptMoE's ACTUAL MLP and PairRouter classes, not a reimplementation.

The upstream file imports SAM1, cv2, and depth-model utilities at module scope.
AST extraction executes only the two unchanged class definitions; the original
file and commit are archived separately. Our supervision/selection are BraTS
adaptations because the authors did not release their training loop.
"""
import ast
from pathlib import Path
import torch
from torch import nn
import numpy as np


def upstream_router(source, cfg):
    tree = ast.parse(Path(source).read_text())
    definitions = [x for x in tree.body if isinstance(x, ast.ClassDef) and x.name in ('MLP', 'PairRouter')]
    if {x.name for x in definitions} != {'MLP', 'PairRouter'}:
        raise ValueError('Expected the archived upstream PromptMoE MLP and PairRouter')
    namespace = {'torch': torch, 'nn': nn}
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(source), 'exec'), namespace)
    return namespace['PairRouter'](cfg)


def possible_pairs(actions):
    # Fixed same-head pairs only. Order is positive then negative. The source
    # router is symmetric; opposite execution orders are NOT conflated.
    return [(i, i+1) for i in (0, 2, 4) if actions[i] is not None and actions[i+1] is not None]


def choose(U, S, actions, *, remaining, margin_pp, pairs=True):
    options = [(float(U[i]), (i,)) for i, action in enumerate(actions) if action is not None]
    if pairs and remaining >= 2:
        options += [(float(S[i, j]), (i, j)) for i, j in possible_pairs(actions)]
    options = [(score, ids) for score, ids in options if np.isfinite(score)]
    if not options:
        return (), 0.0
    score, indices = max(options, key=lambda x: (x[0], -len(x[1]), tuple(-i for i in x[1])))
    # STOP's known utility is exactly zero. margin is validated, not a theorem
    # that the regressed utility is calibrated or positive on unseen patients.
    return (indices, score) if score > margin_pp else ((), score)


def regression_loss(router, context, utility, valid, pairs, pair_valid):
    U, _, S = router.forward_scores(context)
    if not valid.any():
        raise ValueError('No measured candidate utilities')
    loss = nn.functional.smooth_l1_loss(U[valid], utility[valid], beta=1.0)
    if pair_valid.any():
        loss = loss + nn.functional.smooth_l1_loss(S[pair_valid], pairs[pair_valid], beta=1.0)
    return loss
