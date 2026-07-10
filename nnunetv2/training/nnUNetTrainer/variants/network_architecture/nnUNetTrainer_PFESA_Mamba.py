import pydoc

import torch
import torch.nn as nn

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.plans_handling.plans_handler import (
    PlansManager, ConfigurationManager,
)
from nnunetv2.utilities.pfesa_mamba_unet import PFESA_Mamba_PlainConvUNet



class nnUNetTrainer_PFESA_Mamba(nnUNetTrainer):
    """
    PlainConvUNet (2D) + post-conv BiMamba in deepest 2 encoder stages
    + Mamba x2 on bottleneck + PFESA on skip connections.

    Matches the architecture diagram (mamba_bottleneck_architecture.png).
    """

    # ---- knobs (override by subclassing if you want to ablate) ----
    PFESA_RATIO:         float = 0.1
    N_MAMBA_STAGES:      int   = 2   # deepest k encoder stages
    N_BOTTLENECK_BLOCKS: int   = 2   # Mamba x N on bottleneck output
    MAMBA_D_STATE:       int   = 16
    MAMBA_D_CONV:        int   = 4
    MAMBA_EXPAND:        int   = 2

    @staticmethod
    def build_network_architecture(
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

        return PFESA_Mamba_PlainConvUNet(
            input_channels=num_input_channels,
            num_classes=num_output_channels,
            deep_supervision=enable_deep_supervision,
            pfesa_base_ratio=nnUNetTrainer_PFESA_Mamba.PFESA_RATIO,
            n_mamba_stages=nnUNetTrainer_PFESA_Mamba.N_MAMBA_STAGES,
            n_bottleneck_blocks=nnUNetTrainer_PFESA_Mamba.N_BOTTLENECK_BLOCKS,
            mamba_d_state=nnUNetTrainer_PFESA_Mamba.MAMBA_D_STATE,
            mamba_d_conv=nnUNetTrainer_PFESA_Mamba.MAMBA_D_CONV,
            mamba_expand=nnUNetTrainer_PFESA_Mamba.MAMBA_EXPAND,
            **arch_kwargs,
        )

    def _get_mamba_net(self):
        """Return the underlying PFESA_Mamba_PlainConvUNet (peeling DDP /
        torch.compile wrappers if present), or None if the network is
        something else."""
        net = self.network
        # DDP
        if hasattr(net, 'module'):
            net = net.module
        # torch.compile (OptimizedModule)
        if hasattr(net, '_orig_mod'):
            net = net._orig_mod
        return net if isinstance(net, PFESA_Mamba_PlainConvUNet) else None

    def on_train_epoch_end(self, train_outputs):
        super().on_train_epoch_end(train_outputs)

        net = self._get_mamba_net()
        if net is None:
            return

        # --- gamma + norms recorded during the LAST forward of the epoch ---
        rows = []
        for name, blk in net.named_mamba_blocks():
            g  = blk.gamma.detach().float().item()
            xn = blk.last_x_norm.item()
            yn = blk.last_y_norm.item()
            dn = blk.last_delta_norm.item()       # = |gamma * y|
            ratio = (dn / xn) if xn > 0 else 0.0
            rows.append(
                f"  {name:>5s}: gamma={g:+.4e}  "
                f"|x|={xn:.3e}  |y|={yn:.3e}  "
                f"|gamma*y|={dn:.3e}  |gamma*y|/|x|={ratio:.2e}"
            )
        self.print_to_log_file("Mamba diagnostics (last forward of epoch):")
        for r in rows:
            self.print_to_log_file(r)

        grad_total = 0.0
        grad_count = 0
        gamma_grad_total = 0.0
        for _, blk in net.named_mamba_blocks():
            for pname, p in blk.named_parameters():
                if p.grad is None:
                    continue
                g_abs = p.grad.detach().float().abs().sum().item()
                grad_total += g_abs
                grad_count += p.numel()
                if pname == 'gamma':
                    gamma_grad_total += g_abs
        self.print_to_log_file(
            f"Mamba grad check: sum|grad|={grad_total:.3e} over "
            f"{grad_count} params, sum|grad(gamma)|={gamma_grad_total:.3e}"
        )


class nnUNetTrainer_PFESA_Mamba_250epochs(nnUNetTrainer_PFESA_Mamba):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 250

