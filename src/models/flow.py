"""Normalizing Flow (Masked Autoregressive Flow) for helpfulness modeling."""
import torch
import torch.nn as nn
from nflows.flows import Flow
from nflows.distributions import StandardNormal
from nflows.transforms import CompositeTransform, MaskedAffineAutoregressiveTransform, RandomPermutation
#  Normalizing Flow Architecture (MAF)
def create_flow(features=1, context_features=32, num_layers=8, hidden_features=64):
    # Create a Masked Autoregressive Flow for conditional density estimation.
    base_dist = StandardNormal(shape=[features])
    transform_list = []
    for _ in range(num_layers):
        transform_list.append(MaskedAffineAutoregressiveTransform(
            features=features,
            hidden_features=hidden_features,
            context_features=context_features,
            num_blocks=2
        ))
        transform_list.append(RandomPermutation(features=features))
    transform = CompositeTransform(transform_list)
    return Flow(transform, base_dist)
