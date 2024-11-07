import torch

def augment_data(data, device):
    B, M, T = data.shape
    augmented_data = torch.zeros_like(data)

    for i in range(B):
        acc = data[i, :, :]

        # Random channel swapping
        permuted_indices = torch.randperm(M)  # Get a random permutation of channels
        acc = acc[permuted_indices, :]

        # Independent mirroring for each channel
        for j in range(M):
            if torch.rand(1).item() > 0.5:  # 50% chance to mirror each channel independently
                acc[j, :] = 1 - acc[j, :]  # Mirror by flipping around y = 0.5

        # Assign to augmented_data
        augmented_data[i, :, :] = acc

    return augmented_data
