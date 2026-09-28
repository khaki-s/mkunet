import math
import pydoc

import numpy as np
import torch
import torch.nn as nn

from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.loss.monai_region_loss import MonaiRegionTverskyBCE
from nnunetv2.utilities.FADSR.loss import DC_and_BCE_loss_weighted

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler
from nnunetv2.utilities.plans_handling.plans_handler import (
    PlansManager,
    ConfigurationManager,
)
from nnunetv2.utilities.fadsr_unet import FADSR_PlainConvUNet
from nnunetv2.utilities.FADSR.diagnostics import (
    BranchDiagnostics, split_no_decay_params,
)

def _subband_names(n_sub):
    ndim = max(1, int(round(math.log2(max(n_sub, 1)))))
    out = []
    for i in range(n_sub):
        out.append(''.join('H' if (i >> (ndim - 1 - j)) & 1 else 'L'
                           for j in range(ndim)))
    return out


def _unwrap(trainer):
    net = trainer.network.module if trainer.is_ddp else trainer.network
    return getattr(net, '_orig_mod', net)          # torch.compile unwrap


class BranchDiagnosticsMixin:
    enable_branch_diagnostics: bool = True
    diag_grad_every: int = 200

    def _do_i_compile(self):
        if self.enable_branch_diagnostics:
            return False
        return super()._do_i_compile()

    def on_train_start(self):
        super().on_train_start()
        self._diag = None
        if not self.enable_branch_diagnostics:
            return
        net = _unwrap(self)
        self._diag = BranchDiagnostics(net, grad_every=self.diag_grad_every)
        self.print_to_log_file(
            f'BranchDiagnostics: tracking {len(self._diag._targets)} gated '
            f'branches (torch.compile forced off so forward hooks fire)')

    def train_step(self, batch: dict) -> dict:
        out = super().train_step(batch)
        if getattr(self, '_diag', None) is not None:
            self._diag.sample_grads()
        return out

    def on_train_epoch_end(self, train_outputs):
        super().on_train_epoch_end(train_outputs)
        if getattr(self, '_diag', None) is not None:
            self.print_to_log_file(self._diag.format_report(), also_print_to_console=False)
            self._diag.reset()


class WaveletGateLogMixin:
    wavelet_gate_log_every: int = 1

    def on_train_start(self):
        super().on_train_start()
        net = _unwrap(self)
        cfg = getattr(net, 'fadsr_config', None)
        if cfg is not None:
            self.print_to_log_file(f'FADSR config: {cfg}')
        self._log_reconstruction_error(net)

    def _log_reconstruction_error(self, net):
        mods = getattr(net, 'fadsr_modules', None)
        if not mods:
            return
        was_training = net.training
        net.eval()
        try:
            lines = []
            for d, m in enumerate(mods):
                # big enough that every configured level is exercised; 3D probes
                # are kept smaller so this stays cheap on the deepest stages
                side = 32 if getattr(m, 'ndim', 2) == 2 else 16
                x = torch.randn(1, m.enc_dim, *([side] * getattr(m, 'ndim', 2)),
                                device=self.device)
                err = m.reconstruction_error(x)
                flag = 'OK' if err < 1e-3 else 'FAIL -- transform does not invert'
                lines.append(f'  stage {d}: rel_l2={err:.3e}  [{flag}]')
            self.print_to_log_file(
                'FADSR DWT/IDWT round-trip @init (mixers bypassed; expect ~1e-7):\n'
                + '\n'.join(lines))
        finally:
            net.train(was_training)

    def on_train_epoch_end(self, train_outputs):
        super().on_train_epoch_end(train_outputs)
        if (self.current_epoch + 1) % self.wavelet_gate_log_every:
            return
        net = _unwrap(self)
        if not hasattr(net, 'fadsr_gate_report'):
            return
        rep = net.fadsr_gate_report()
        n_sub = len(next(iter(rep.values()))[0]) if rep else 0
        names = _subband_names(n_sub)
        lines = ['WaveletGates  effective gain = mean_c|scale| * |band_gate|',
                 f'{"stage/level":<12} '
                 + ' '.join(f'{nm:>9}' for nm in names)
                 + f' {"HF/approx":>10}']
        for d, levels in rep.items():
            for k, bands in enumerate(levels):
                approx, hf_bands = bands[0], bands[1:]
                hf = sum(hf_bands) / len(hf_bands) if hf_bands else 0.0
                ratio = hf / approx if approx > 0 else float('inf')
                lines.append(f'{f"s{d}/L{k}":<12} '
                             + ' '.join(f'{v:9.5f}' for v in bands)
                             + f' {ratio:10.3f}')
        self.print_to_log_file('\n'.join(lines), also_print_to_console=False)

        llr = net.fadsr_ll_report() if hasattr(net, 'fadsr_ll_report') else {}
        if llr:
            self.print_to_log_file(
                'LLEnhance mean |gate| per stage: '
                + '  '.join(f's{d}={v:.5f}' for d, v in llr.items()),
                also_print_to_console=False)

        shr = (net.fadsr_shrink_report()
               if hasattr(net, 'fadsr_shrink_report') else {})
        if shr:
            n_hf = len(next(iter(shr.values()))[0])
            hf_names = _subband_names(n_hf + 1)[1:]
            sl = ['HFShrink  mean soft-threshold (0 = identity, growing = it is '
                  'actually suppressing HF)',
                  f'{"stage/level":<12} ' + ' '.join(f'{nm:>9}' for nm in hf_names)]
            for d, levels in shr.items():
                for k, bands in enumerate(levels):
                    sl.append(f'{f"s{d}/L{k}":<12} '
                              + ' '.join(f'{v:9.5f}' for v in bands))
            self.print_to_log_file('\n'.join(sl), also_print_to_console=False)


class NoDecayGatesMixin:
    def configure_optimizers(self):
        decay, no_decay = split_no_decay_params(
            self.network, extra_names=('scale', 'gamma', 'base_scale', 'band_gate'))
        optimizer = torch.optim.SGD(
            [{'params': decay, 'weight_decay': self.weight_decay},
             {'params': no_decay, 'weight_decay': 0.0}],
            self.initial_lr, momentum=0.99, nesterov=True)
        self.print_to_log_file(
            f'optimizer: {len(decay)} tensors with wd={self.weight_decay}, '
            f'{len(no_decay)} gate/LayerScale tensors with wd=0')
        return optimizer, PolyLRScheduler(optimizer, self.initial_lr, self.num_epochs)

class nnUNetTrainer_FADSR(nnUNetTrainer):
    # subclasses set this to rewire FADSR without touching the plans
    fadsr_arch_overrides: dict = {}

    @classmethod
    def build_network_architecture(
        cls,
        plans_manager: PlansManager,
        configuration_manager: ConfigurationManager,
        num_input_channels: int,
        num_output_channels: int,
        enable_deep_supervision: bool = True,
    ) -> nn.Module:
        arch_kwargs = dict(**configuration_manager.network_arch_init_kwargs)
        for ri in configuration_manager.network_arch_init_kwargs_req_import:
            if arch_kwargs[ri] is not None:
                arch_kwargs[ri] = pydoc.locate(arch_kwargs[ri])
        arch_kwargs.update(cls.fadsr_arch_overrides)
        return FADSR_PlainConvUNet(
            input_channels=num_input_channels,
            num_classes=num_output_channels,
            deep_supervision=enable_deep_supervision,
            **arch_kwargs
        )

_DB1_PARITY = dict(
    fadsr_wt_type='db1', fadsr_dw_kernel=5, fadsr_mixer_dilations=(1, 1, 1),
    fadsr_subband_layout='cat', fadsr_mixer_scale=0.0, fadsr_min_level_size=1,
    use_bottleneck_attn=False, use_spatial_block=False,
)


class nnUNetTrainer_FADSR_A_db1parity_100epochs(BranchDiagnosticsMixin,
                                                WaveletGateLogMixin,
                                                NoDecayGatesMixin,
                                                nnUNetTrainer_FADSR):
    """Control. Should reproduce 91.11/91.70/88.92."""
    fadsr_arch_overrides = dict(_DB1_PARITY)

    def __init__(self, plans, configuration, fold, dataset_json,
                 device=torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 100


class nnUNetTrainer_FADSR_B_db2_100epochs(nnUNetTrainer_FADSR_A_db1parity_100epochs):
    """A + db2. Isolates 'does a longer wavelet help now that it inverts'."""
    fadsr_arch_overrides = {**_DB1_PARITY, 'fadsr_wt_type': 'db2','dw_kernel':7}


class nnUNetTrainer_FADSR_C_stack_100epochs(nnUNetTrainer_FADSR_A_db1parity_100epochs):
    """B + channel-major layout. Isolates real inter-subband mixing."""
    fadsr_arch_overrides = {**_DB1_PARITY, 'fadsr_wt_type': 'db2',
                            'fadsr_subband_layout': 'stack'}


class nnUNetTrainer_FADSR_D_k7dil_100epochs(nnUNetTrainer_FADSR_A_db1parity_100epochs):
    """C + k=7 and dilation on the high-res levels, and the level cap back on."""
    fadsr_arch_overrides = {**_DB1_PARITY, 'fadsr_wt_type': 'db2',
                            'fadsr_subband_layout': 'stack',
                            'fadsr_dw_kernel': 7,
                            'fadsr_mixer_dilations': (2, 1, 1),
                            'fadsr_min_level_size': 4}


class nnUNetTrainer_FADSR_E_warmgate_100epochs(nnUNetTrainer_FADSR_A_db1parity_100epochs):
    """Best-so-far + gate init 1e-2 instead of 0.

    At scale exactly 0, dL/dW = dL/dy * scale * (...) = 0, so the branch weights
    cannot learn until `scale` alone drags itself off 0 -- db1 needed 100 epochs
    to reach 0.009. 1e-2 is ~0.34% perturbation, i.e. where db1 ended up anyway,
    but both the gate and the branch weights get gradient from step 0.
    """
    fadsr_arch_overrides = {**_DB1_PARITY, 'fadsr_wt_type': 'db2',
                            'fadsr_subband_layout': 'stack',
                            'fadsr_mixer_scale': 1e-2}


class nnUNetTrainer_FADSR_J_llenh_100epochs(nnUNetTrainer_FADSR_A_db1parity_100epochs):
    """A + LLEnhance on the deepest LL. No HF change, no ASPP.

    Follows the direction all three gate logs already point: LL gate ~0.03 vs
    HF/LL ~0.09-0.18, and H proved suppressing HF costs WT. So leave HF alone and
    give LL a dedicated branch instead. See LLEnhance for the full argument.
    """
    fadsr_arch_overrides = {**_DB1_PARITY, 'fadsr_ll_enhance': True}


class TverskyRegionLossMixin:
    tversky_alphas = (0.5, 0.3, 0.5)      # FP weight, ordered (WT, TC, ET)
    tversky_betas = (0.5, 0.7, 0.5)       # FN weight; beta > alpha favours recall

    def _build_loss(self):
        if not self.label_manager.has_regions:
            raise RuntimeError('TverskyRegionLossMixin requires region-based '
                               'training; this dataset has_regions=False')
        loss = MonaiRegionTverskyBCE(
            alphas=self.tversky_alphas,
            betas=self.tversky_betas,
            batch_dice=self.configuration_manager.batch_dice,
            smooth=1e-5,
        )
        self.print_to_log_file(
            f'loss: MonaiRegionTverskyBCE alphas(FP)={tuple(self.tversky_alphas)} '
            f'betas(FN)={tuple(self.tversky_betas)} '
            f'(alpha=beta=0.5 would be plain Dice)')

        if self.enable_deep_supervision:
            scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2 ** i) for i in range(len(scales))])
            weights[-1] = 0
            weights = weights / weights.sum()
            loss = DeepSupervisionWrapper(loss, weights)
        return loss


class nnUNetTrainer_FADSR_N_tversky_100epochs(TverskyRegionLossMixin,
                                             nnUNetTrainer_FADSR_A_db1parity_100epochs):
    """A + Tversky beta>alpha. Architecture byte-identical to A."""
    tversky_alphas = (0.5, 0.3, 0.5)      # FP weight, ordered (WT, TC, ET)
    tversky_betas = (0.5, 0.7, 0.5)  


class nnUNetTrainer_FADSR_Ndice_control_100epochs(nnUNetTrainer_FADSR_N_tversky_100epochs):
    """alpha=beta=0.5, i.e. plain Dice via the Tversky path.

    The control that makes N interpretable: it must land on A. If it does not, the
    loss swap changed something other than the FP/FN price and N cannot be read.
    Also doubles as the repeated-A run that finally measures the noise floor.
    """
    tversky_alphas = (0.5, 0.5, 0.5)
    tversky_betas = (0.5, 0.5, 0.5)


class nnUNetTrainer_FADSR_N2_tversky_100epochs(nnUNetTrainer_FADSR_N_tversky_100epochs):
    """Milder: alpha=0.4/beta=0.6. Use if N overshoots into false positives."""
    tversky_alphas = (0.4, 0.4, 0.4)
    tversky_betas = (0.6, 0.6, 0.6)


class nnUNetTrainer_FADSR_L_k7_100epochs(nnUNetTrainer_FADSR_A_db1parity_100epochs):
    """A + dw_kernel 7 everywhere. Single knob (kernel size does not shift RNG
    for anything built earlier, and the mixer is the last thing built).

    Included for completeness, but the tap measurement argues against it: at
    stage 3/4 a k=7 subband conv is 82%/92% padding at L2. See L2 below.
    """
    fadsr_arch_overrides = {**_DB1_PARITY, 'fadsr_dw_kernel': 7}


class nnUNetTrainer_FADSR_M_perlevelk_100epochs(nnUNetTrainer_FADSR_A_db1parity_100epochs):
    """A + per-level kernels 7/5/3, coarse-to-fine.

    This is the version of "bigger kernel" the measurement actually supports:
    k=7 only at L0 where the subband is still large (96x80 at stage 0, 96% of taps
    in bounds), k=3 at L2 where even k=5 is 64-84% padding at the deeper stages.
    Gets the extra receptive field where it is usable and stops paying for it
    where it is not.
    """
    fadsr_arch_overrides = {**_DB1_PARITY, 'fadsr_dw_kernels': (7, 5, 3)}


class nnUNetTrainer_FADSR_H_shrink_100epochs(nnUNetTrainer_FADSR_A_db1parity_100epochs):
    """A + learnable HF soft-thresholding. No ASPP.

    Single knob from A, and unlike ASPP it targets the actual residual: PP lifts
    WT 91.02 -> 91.21 by deleting isolated FP components, i.e. small isolated HF
    energy, which is precisely what shrinkage suppresses. See HFShrink for why
    this cannot deadlock the way the ASPP gamma did.

    Adds 3 * enc_dim params per level -- ~5.8k total, vs ~31k for ASPP.
    """
    fadsr_arch_overrides = {**_DB1_PARITY, 'fadsr_hf_shrink': True}


class nnUNetTrainer_FADSR_A_ft50_100epochs(nnUNetTrainer_FADSR_A_db1parity_100epochs):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 50
class nnUNetTrainer_ft50(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 50

class nnUNetTrainer_SSA100eopochs(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 100

class nnUNetTrainer_FADSR_A__SSA100epochs(nnUNetTrainer_FADSR_A_db1parity_100epochs):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 100

_DB1_PARITY_3D = dict(
    fadsr_wt_type='db1', fadsr_dw_kernel=3, fadsr_mixer_dilations=(1, 1, 1),
    fadsr_subband_layout='cat', fadsr_mixer_scale=0.0, fadsr_min_level_size=2,
    use_bottleneck_attn=False, use_spatial_block=False,
)


class nnUNetTrainer_FADSR_3d_200epochs(BranchDiagnosticsMixin,
                             WaveletGateLogMixin,
                             NoDecayGatesMixin,
                             nnUNetTrainer_FADSR):
    fadsr_arch_overrides = dict(_DB1_PARITY_3D)

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 200


class nnUNetTrainer_FADSR_3d_db2(nnUNetTrainer_FADSR_3d_200epochs):
    fadsr_arch_overrides = {**_DB1_PARITY_3D, 'fadsr_wt_type': 'db2'}


class nnUNetTrainer_FADSR_3d_stack(nnUNetTrainer_FADSR_3d_200epochs):
    fadsr_arch_overrides = {**_DB1_PARITY_3D, 'fadsr_subband_layout': 'stack'}


class nnUNetTrainer_FADSR_99epochs(nnUNetTrainer_FADSR):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 99


class nnUNetTrainer_FADSR_100epochs_fixdb2(BranchDiagnosticsMixin,
                                    WaveletGateLogMixin,
                                    NoDecayGatesMixin,
                                    nnUNetTrainer_FADSR):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 100

