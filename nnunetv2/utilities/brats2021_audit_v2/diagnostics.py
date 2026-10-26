"""Bounded-cost diagnostics for the added BraTS modules.

Gradient measurements are AFTER AMP unscale and global gradient clipping,
because the inherited train_step owns those operations. Feature perturbation
is not causal importance. A separate paired eval-patch ablation reports the
effect of bypassing one module in the current trained model, not retraining it.
No tensors with autograd graphs are retained between steps.
"""
import json
import math
import random
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def finite_number(value):
    value = float(value)
    return value if math.isfinite(value) else None


def spatial_sample(x, budget=32768):
    """Strided spatial sample without flattening/copying a full feature volume."""
    step = max(1, math.ceil((x.numel() / budget) ** (1 / max(1, x.ndim - 2))))
    return x[(slice(None), slice(None), *[slice(None, None, step) for _ in x.shape[2:]])].detach().float()


def exported_masks(logits):
    """Match nnU-Net's WT->TC->ET overwrite semantics for integer output."""
    wt, tc, et = (logits.float().sigmoid() > .5).unbind(1)
    return {'WT': wt | tc | et, 'TC': tc | et, 'ET': et,
            'ED': wt & ~tc & ~et, 'NCR': tc & ~et}


def patch_metrics(logits, target):
    masks = exported_masks(logits)
    wt, tc, et = (target > .5).unbind(1)
    refs = {'WT': wt, 'TC': tc, 'ET': et, 'ED': wt & ~tc, 'NCR': tc & ~et}
    metrics = {}
    for name, mask in masks.items():
        ref = refs[name]
        n_ref, n_pred = int(ref.sum()), int(mask.sum())
        tp = int((ref & mask).sum())
        metrics[name] = {'dice': 2 * tp / (n_ref + n_pred) if n_ref + n_pred else None,
                         'n_ref': n_ref, 'n_pred': n_pred, 'fp': n_pred - tp, 'fn': n_ref - tp}
    return metrics


@contextmanager
def preserved_rng(device):
    python_state, numpy_state = random.getstate(), np.random.get_state()
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == 'cuda' else []
    try:
        with torch.random.fork_rng(devices=devices):
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


class ModuleDiagnostics:
    def __init__(self, network):
        self.network = network
        self.modules = {}
        for name in ('modality_gate', 'moe', 'context'):
            module = getattr(network, name, None)
            if isinstance(module, nn.Module) and not isinstance(module, nn.Identity):
                self.modules[name] = module
        for i, module in enumerate(getattr(network, 'frequency', ())):
            self.modules[f'frequency.{i}'] = module
        self.active = False
        self.capture_output = False
        self.full_logits = None
        self.records = defaultdict(list)
        self.before = {}
        self.step_features = {}
        self.handles = [module.register_forward_hook(self._feature_hook(name))
                        for name, module in self.modules.items()]
        self.handles.append(network.register_forward_hook(self._network_hook))

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.before.clear()
        self.full_logits = None

    def _network_hook(self, module, inputs, outputs):
        if self.capture_output:
            self.full_logits = (outputs[0] if isinstance(outputs, (list, tuple)) else outputs).detach()

    def _feature_hook(self, name):
        def hook(module, inputs, output):
            if not self.active:
                return
            with torch.no_grad():
                x, y = spatial_sample(inputs[0]), spatial_sample(output)
                delta = y - x
                row = {'feature_delta_rms_over_input_rms': finite_number(delta.square().mean().sqrt() / (x.square().mean().sqrt() + 1e-8)),
                       'feature_delta_mean_abs': finite_number(delta.abs().mean())}
                if name == 'modality_gate':
                    # Ratio is identifiable only at nonzero sampled input voxels.
                    gains = []
                    for channel in range(x.shape[1]):
                        valid = x[:, channel].abs() > 1e-4
                        g = y[:, channel][valid] / x[:, channel][valid]
                        gains.append({'mean': finite_number(g.mean()) if g.numel() else None,
                                      'fraction_below_055': finite_number((g < .55).float().mean()) if g.numel() else None,
                                      'fraction_above_145': finite_number((g > 1.45).float().mean()) if g.numel() else None})
                    row['gain_T1_T1ce_T2_FLAIR'] = gains
                self.step_features[name] = row
        return hook

    def begin_step(self):
        self.active = True
        self.step_features.clear()
        self.before = {name: [p.detach().clone() for p in module.parameters()]
                       for name, module in self.modules.items()}

    def cancel_step(self):
        self.active = False
        self.before.clear()
        self.step_features.clear()

    def end_step(self, amp_step_skipped=False):
        self.active = False
        with torch.no_grad():
            for name, module in self.modules.items():
                params = list(module.parameters())
                grads = [p.grad.detach().float() for p in params if p.grad is not None]
                grad_sq = sum(float(g.square().sum()) for g in grads)
                before = self.before[name]
                weight_sq = sum(float(p.float().square().sum()) for p in before)
                update_sq = sum(float((p.detach().float() - old.float()).square().sum()) for p, old in zip(params, before))
                grad_elements = sum(g.numel() for g in grads)
                grad_nonzero = sum(int(torch.count_nonzero(g)) for g in grads)
                row = {**self.step_features.get(name, {}),
                       'grad_l2_after_unscale_clip': finite_number(math.sqrt(grad_sq)),
                       'grad_nonzero_fraction': grad_nonzero / grad_elements if grad_elements else None,
                       'grad_none_tensors': sum(p.grad is None for p in params),
                       'parameter_l2_before_step': finite_number(math.sqrt(weight_sq)),
                       'update_l2': finite_number(math.sqrt(update_sq)),
                       'update_over_parameter_l2': finite_number(math.sqrt(update_sq) / (math.sqrt(weight_sq) + 1e-12)),
                       'amp_step_skipped': bool(amp_step_skipped)}
                # Separate router/expert gradients: a live residual scale alone
                # does not imply that the experts or router are learning.
                groups = defaultdict(list)
                for parameter_name, parameter in module.named_parameters():
                    parts = parameter_name.split('.')
                    group = '.'.join(parts[:2]) if parts[0] == 'experts' else parts[0]
                    groups[group].append(parameter)
                row['grad_groups'] = {}
                for group, parameters in groups.items():
                    group_grads = [p.grad.detach().float() for p in parameters if p.grad is not None]
                    elements = sum(g.numel() for g in group_grads)
                    squares = sum(float(g.square().sum()) for g in group_grads)
                    nonzero = sum(int(torch.count_nonzero(g)) for g in group_grads)
                    row['grad_groups'][group] = {
                        'grad_l2_after_unscale_clip': finite_number(math.sqrt(squares)),
                        'grad_rms_after_unscale_clip': finite_number(math.sqrt(squares / elements)) if elements else None,
                        'grad_nonzero_fraction': nonzero / elements if elements else None,
                        'grad_none_tensors': sum(p.grad is None for p in parameters)}
                scale = getattr(module, 'scale', None)
                if scale is not None:
                    row['residual_scale_tanh_mean_abs'] = finite_number(scale.detach().float().tanh().abs().mean())
                self.records[name].append(row)
        self.before.clear()
        self.step_features.clear()

    def epoch_summary(self):
        result = {}
        for name, rows in self.records.items():
            scalar_keys = set().union(*(r.keys() for r in rows)) - {'gain_T1_T1ce_T2_FLAIR', 'grad_groups'}
            summary = {'sampled_steps': len(rows)}
            for key in sorted(scalar_keys):
                values = [r[key] for r in rows if r.get(key) is not None]
                summary[key + '_mean'] = float(np.mean(values)) if values else None
            if any('gain_T1_T1ce_T2_FLAIR' in r for r in rows):
                summary['gain_T1_T1ce_T2_FLAIR'] = []
                for i in range(4):
                    values = [r['gain_T1_T1ce_T2_FLAIR'][i] for r in rows if 'gain_T1_T1ce_T2_FLAIR' in r]
                    summary['gain_T1_T1ce_T2_FLAIR'].append({k: float(np.mean([v[k] for v in values if v[k] is not None])) if any(v[k] is not None for v in values) else None
                                                          for k in ('mean', 'fraction_below_055', 'fraction_above_145')})
            group_names = sorted(set().union(*(r.get('grad_groups', {}).keys() for r in rows)))
            if group_names:
                summary['grad_groups'] = {}
                for group in group_names:
                    group_rows = [r['grad_groups'][group] for r in rows if group in r.get('grad_groups', {})]
                    group_summary = {}
                    for key in sorted(set().union(*(r.keys() for r in group_rows))):
                        values = [r[key] for r in group_rows if r.get(key) is not None]
                        group_summary[key + '_mean'] = float(np.mean(values)) if values else None
                    summary['grad_groups'][group] = group_summary
            result[name] = summary
        self.records.clear()
        return result

    @torch.no_grad()
    def ablate(self, data, target):
        """Full minus bypass Dice: positive means module helps this eval patch."""
        if self.network.training:
            raise RuntimeError('Module ablation must run in eval mode')
        full_logits = self.full_logits
        self.full_logits = None
        if full_logits is None:
            raise RuntimeError('No unmodified validation output was captured')
        full = patch_metrics(full_logits, target)
        full_bce = F.binary_cross_entropy_with_logits(full_logits.float(), target.float())
        result = {'scope': 'one current validation batch; patch-pooled, not case/sliding-window Dice',
                  'dice_delta_sign': 'full_minus_bypass_positive_means_helpful',
                  'full': full, 'modules': {}}
        with preserved_rng(data.device):
            for name, module in self.modules.items():
                handle = module.register_forward_hook(lambda mod, inputs, output: inputs[0])
                try:
                    with torch.autocast(device_type=data.device.type, enabled=data.device.type == 'cuda'):
                        out = self.network(data)
                    logits = out[0] if isinstance(out, (list, tuple)) else out
                    bypass = patch_metrics(logits, target)
                    bce = F.binary_cross_entropy_with_logits(logits.float(), target.float())
                    result['modules'][name] = {
                        'bypass': bypass,
                        'delta_dice_pp_full_minus_bypass': {r: 100 * (full[r]['dice'] - bypass[r]['dice']) if full[r]['dice'] is not None and bypass[r]['dice'] is not None else None for r in full},
                        'delta_bce_bypass_minus_full': finite_number(bce - full_bce),
                        'mean_abs_region_probability_change': finite_number((logits.float().sigmoid() - full_logits.float().sigmoid()).abs().mean())}
                    del out, logits
                finally:
                    handle.remove()
        return result


class ModuleDiagnosticsMixin:
    """Hook into inherited steps, preserving optimizer/loss/checkpoint behavior."""
    def _get_module_diagnostics(self):
        if not hasattr(self, '_module_diag'):
            import os
            self._diag_every = int(os.environ.get('BRATS_DIAG_EVERY', '50'))
            self._ablation_every = int(os.environ.get('BRATS_ABLATE_EVERY', '25'))
            if self._diag_every < 0 or self._ablation_every < 0:
                raise ValueError('BRATS_DIAG_EVERY and BRATS_ABLATE_EVERY must be nonnegative; 0 disables')
            self._module_diag = ModuleDiagnostics(self.network) if self._diag_every or self._ablation_every else None
            self._diag_steps = 0
            self._ablation_done_epoch = None
            self.print_to_log_file('MODULE_DIAGNOSTICS_CONFIG', {
                'train_every_steps': self._diag_every, 'ablate_every_epochs': self._ablation_every,
                'ablate_also_first_epoch': bool(self._ablation_every),
                'gradient_stage': 'after AMP unscale and global clipping',
                'feature_change_is_not_causal_contribution': True})
        return self._module_diag

    def _write_module_record(self, kind, payload):
        record = {'kind': kind, 'epoch': int(self.current_epoch), 'epoch_1based': int(self.current_epoch) + 1,
                  'trainer': type(self).__name__, **payload}
        encoded = json.dumps(record, ensure_ascii=False, allow_nan=False)
        self.print_to_log_file(kind, encoded)
        with (Path(self.output_folder) / 'module_diagnostics.jsonl').open('a') as stream:
            stream.write(encoded + '\n')

    def train_step(self, batch):
        monitor = self._get_module_diagnostics()
        sampled = monitor is not None and monitor.modules and self._diag_every > 0 and self._diag_steps % self._diag_every == 0
        if sampled:
            monitor.begin_step()
        old_scale = self.grad_scaler.get_scale() if sampled and self.grad_scaler is not None else None
        try:
            result = super().train_step(batch)
        except BaseException:
            if sampled:
                monitor.cancel_step()
            raise
        if sampled:
            new_scale = self.grad_scaler.get_scale() if self.grad_scaler is not None else None
            monitor.end_step(old_scale is not None and new_scale < old_scale)
        self._diag_steps += 1
        return result

    def on_train_epoch_end(self, train_outputs):
        super().on_train_epoch_end(train_outputs)
        monitor = getattr(self, '_module_diag', None)
        if monitor is not None:
            summary = monitor.epoch_summary()
            if summary:
                self._write_module_record('MODULE_GRAD_FEATURE', {'modules': summary})

    def validation_step(self, batch):
        monitor = self._get_module_diagnostics()
        epoch = int(self.current_epoch)
        probe = monitor is not None and monitor.modules and self._ablation_every > 0 and self._ablation_done_epoch != epoch and (epoch == 0 or (epoch + 1) % self._ablation_every == 0)
        if not probe:
            return super().validation_step(batch)
        monitor.capture_output = True
        try:
            result = super().validation_step(batch)
        except BaseException:
            monitor.full_logits = None
            raise
        finally:
            monitor.capture_output = False
        try:
            target = batch['target'][0] if isinstance(batch['target'], (list, tuple)) else batch['target']
            payload = monitor.ablate(batch['data'].to(self.device, non_blocking=True), target.to(self.device, non_blocking=True))
            if 'keys' in batch:
                payload['case_keys'] = [str(k) for k in batch['keys']]
            self._write_module_record('MODULE_ABLATION', payload)
            self._ablation_done_epoch = epoch
        finally:
            monitor.full_logits = None
        return result

    def on_train_end(self):
        try:
            return super().on_train_end()
        finally:
            monitor = getattr(self, '_module_diag', None)
            if monitor is not None:
                monitor.close()
