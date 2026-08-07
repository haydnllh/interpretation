import numpy as np

def gaussian_RBF(X, X_samples, kernel_width):
    """Generate weights based on the Gaussian RBF kernel"""
    distance = np.linalg.norm(X_samples - X, axis=1)
    weights = np.exp(- distance / kernel_width)
    return weights