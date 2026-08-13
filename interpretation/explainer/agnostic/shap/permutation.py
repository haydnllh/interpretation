import numpy as np

def generate_permutations(
    X,
    samples,
    n_permutations,
):
    """
    Generate the feature permutation used for estimating Shapley values
    Returns data for inference with shape (n_permutations, n_features, n_samples, n_features)
    and the permutation indices (n_permutations, n_features)
    """
    n_features = len(X)
    
    perms = np.array([np.random.permutation(n_features) for _ in range(n_permutations)])
    
    identity = np.eye(n_features, dtype=bool)
    masks = np.cumsum(identity[perms], axis=1).astype(bool)
    
    result = np.tile(samples, (n_permutations, n_features, 1, 1))
    
    mask_expanded = masks[:, :, np.newaxis, :]
    
    result = np.where(mask_expanded, X, result)
    
    return result, perms