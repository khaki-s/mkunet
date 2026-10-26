"""Nonnegative direct refined-agent supervision; equal weight per patient."""
from ..agent_v1.losses import positive_region_loss, present_expert_loss


def refined_auxiliary_loss(experts, target, mask):
    # The shared trainer rejects ignore labels and uses three nested region targets.
    return present_expert_loss(experts, target, mask, positive_region_loss)


def refined_weight(epoch, maximum=.1, start=100, ramp_epochs=20):
    if maximum < 0 or ramp_epochs < 2:
        raise ValueError('Expected nonnegative weight and at least two ramp epochs')
    return float(maximum) * min(1., max(0., (int(epoch) - start) / (ramp_epochs - 1)))
