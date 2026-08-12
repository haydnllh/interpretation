import numpy as np

def generate_permutations(
    X,
    samples,
    n_permutation,
):
    n_features = len(X)
    
    perms = np.array([np.random.permutation(n_features) for _ in range(n_permutation)])
    
    identity = np.eye(n_features, dtype=bool)
    masks = np.cumsum(identity[perms], axis=1).astype(bool)
    
    result = np.tile(samples, (n_permutation, n_features, 1, 1))
    
    mask_expanded = masks[:, :, np.newaxis, :]
    
    result = np.where(mask_expanded, X, result)
    
    return result, perms