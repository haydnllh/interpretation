import numpy as np
import numpy.typing as npt
from typing import Callable
from ....models import NNModel

def smooth_grad(
    model: NNModel,
    X: npt.NDArray,
    class_idx: int,
    objective_fn: Callable,
    n_samples: int = 50,
):
    sigma = 0.15 * (np.max(X) - np.min(X))
    
    grad_acc = np.zeros_like(X)
    
    for _ in range(n_samples):
        noise = np.random.normal(0, sigma, size=X.shape)
        X_sample = X + noise
        
        grad_acc += model.compute_gradients(
            X_sample,
            objective=(-1, class_idx),
            objective_fn=objective_fn,
        )
    
    return grad_acc / n_samples    