"""Training-time instrumentation for gated residual branches.

Every added module in this codebase has the shape

    y = x + scale * branch(x)

so its influence on the network is `scale * ||branch(x)|| / ||x||`, not the
branch's parameter count. Measured on the trained checkpoints, that number came
out at <1% for FADSR and ~0.007% for BottleneckASPPCBAM -- i.e. the modules
were nearly inert and the architecture comparisons were mostly noise. This
module reports it *during* training so that failure mode is visible on epoch 1
instead of after a 100-epoch run.

Two independent things get logged, and they answer different questions:

  contrib%  = ||y - x|| / ||x||   (forward hook, on real batches)
              How much the branch actually changes the signal. This is the one
              that matters for "is this module doing anything".

  grad ratio = ||dL/d(branch weights)|| / ||dL/d(scale)||
              Whether the branch weights are being trained at all. At scale=0
              this is ~0 by construction: dL/dW = dL/dy * scale * (...) = 0, so
              only `scale` moves and the branch has to wait for it to leave 0.
              A healthy branch has this well above zero from step 0.

Cheap by design: forward hooks compute two norms per module per step, gradient
stats are sampled every `grad_every` steps.
"""

import torch
import torch.nn as nn


class BranchDiagnostics:
    def __init__(self, network, grad_every=50):
        self.network = network
        self.grad_every = int(grad_every)
        self._handles = []
        self._fwd = {}          # name -> [sum_ratio, n]
        self._grad = {}         # name -> [sum_ratio, sum_gnorm_branch, sum_gnorm_gate, n]
        self._step = 0
        self._targets = self._find_targets()
        self._register()

    # ---- discovery -------------------------------------------------------
    def _find_targets(self):
        """Modules of the form y = x + scale * branch(x), by gate param name."""
        out = []
        for name, mod in self.network.named_modules():
            gate = None
            for pname, p in mod.named_parameters(recurse=False):
                if pname in ('scale', 'gamma', 'base_scale'):
                    gate = pname
                    break
            if gate is not None:
                out.append((name, mod, gate))
        return out

    def _register(self):
        for name, mod, _ in self._targets:
            self._handles.append(mod.register_forward_hook(self._make_fwd_hook(name)))

    def _make_fwd_hook(self, name):
        def hook(mod, inputs, output):
            if not mod.training or not torch.is_tensor(output):
                return
            x = inputs[0] if inputs and torch.is_tensor(inputs[0]) else None
            if x is None or x.shape != output.shape:
                return
            with torch.no_grad():
                nx = x.norm()
                if nx <= 0:
                    return
                ratio = ((output - x).norm() / nx).item()
            slot = self._fwd.setdefault(name, [0.0, 0])
            slot[0] += ratio
            slot[1] += 1
        return hook

    # ---- gradient sampling ----------------------------------------------
    def sample_grads(self):
        """Call right after backward, before zero_grad. Grads must be unscaled."""
        self._step += 1
        if self._step % self.grad_every:
            return
        for name, mod, gate in self._targets:
            gp = getattr(mod, gate)
            if gp.grad is None:
                continue
            g_gate = gp.grad.norm().item()
            g_branch = 0.0
            for pname, p in mod.named_parameters():
                if p.grad is None or pname.endswith(gate):
                    continue
                g_branch += p.grad.norm().item() ** 2
            g_branch = g_branch ** 0.5
            slot = self._grad.setdefault(name, [0.0, 0.0, 0.0, 0])
            # gradS is legitimately 0 for a zero-init base branch
            # (dL/d(base_scale) = dL/dy * base_conv(x) and base_conv(x) == 0),
            # so report the ratio as inf rather than a huge 1/eps number.
            slot[0] += (g_branch / g_gate) if g_gate > 0 else float('inf')
            slot[1] += g_branch
            slot[2] += g_gate
            slot[3] += 1

    # ---- reporting -------------------------------------------------------
    def report(self, max_rows=None):
        rows = []
        for name, mod, gate in self._targets:
            gp = getattr(mod, gate)
            f = self._fwd.get(name)
            g = self._grad.get(name)
            rows.append(dict(
                name=name,
                gate=gate,
                gate_mean=gp.detach().abs().mean().item(),
                contrib=(f[0] / f[1]) if f and f[1] else float('nan'),
                grad_ratio=(g[0] / g[3]) if g and g[3] else float('nan'),
                g_branch=(g[1] / g[3]) if g and g[3] else float('nan'),
                g_gate=(g[2] / g[3]) if g and g[3] else float('nan'),
            ))
        if max_rows is not None:
            rows = rows[:max_rows]
        return rows

    def format_report(self, max_rows=None):
        rows = self.report(max_rows)
        if not rows:
            return 'BranchDiagnostics: no gated modules found'
        w = max(len(r['name']) for r in rows)
        head = (f'{"module":<{w}}  {"gate":<10} {"|gate|":>9} {"contrib%":>9} '
                f'{"gradW/gradS":>12} {"|gradW|":>10} {"|gradS|":>10}')
        lines = ['BranchDiagnostics  (contrib% = ||y-x||/||x|| on real batches)', head,
                 '-' * len(head)]

        def num(v, fmt, width):
            if v != v:                      # NaN -> module never ran
                return f'{"unused":>{width}}'
            if v == float('inf'):
                return f'{"inf":>{width}}'
            return f'{v:{fmt}}'

        for r in rows:
            lines.append(
                f'{r["name"]:<{w}}  {r["gate"]:<10} {r["gate_mean"]:9.5f} '
                f'{num(100 * r["contrib"], "9.4f", 9)} '
                f'{num(r["grad_ratio"], "12.3e", 12)} '
                f'{num(r["g_branch"], "10.3e", 10)} '
                f'{num(r["g_gate"], "10.3e", 10)}')
        return '\n'.join(lines)

    def reset(self):
        self._fwd.clear()
        self._grad.clear()

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()


def split_no_decay_params(network, extra_names=('scale', 'gamma', 'base_scale')):
    """Split params into (decay, no_decay). Gates/LayerScale must not decay.

    FADSR.no_weight_decay() exists but nnUNetTrainer.configure_optimizers passes
    self.network.parameters() straight to SGD, so it is never consulted -- and it
    only matched 'scale', missing every LayerScale `gamma`. Matching by leaf name
    here covers both without depending on that method.

    Weight decay on a gate is not harmless: at wd=3e-5 with momentum 0.99 under
    the poly LR schedule, 100 epochs x 250 iters shrinks a gate by ~0.67x on top
    of its already-slow climb out of 0.
    """
    decay, no_decay = [], []
    for name, p in network.named_parameters():
        if not p.requires_grad:
            continue
        leaf = name.rsplit('.', 1)[-1]
        (no_decay if leaf in extra_names else decay).append(p)
    return decay, no_decay
