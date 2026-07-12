# Normalizing Flow (MAF) Architecture
# Extracted from OmniReview_Colab.ipynb

#  Normalizing Flow Architecture (MAF)
def create_flow(num_features=1, num_context=16, num_layers=8):
    # Create a Masked Autoregressive Flow for conditional density estimation.
    base_dist = StandardNormal(shape=[num_features])
    transform_list = []
    for _ in range(num_layers):
        transform_list.append(MaskedAffineAutoregressiveTransform(
            features=num_features,
            hidden_features=32,
            context_features=num_context,
            num_blocks=2
        ))
        transform_list.append(RandomPermutation(features=num_features))
    transform = CompositeTransform(transform_list)
    return flows.Flow(transform, base_dist)

print(" Normalizing Flow architecture defined")

