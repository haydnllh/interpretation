from ..nn_explainer import NNExplainer
import numpy as np
import numpy.typing as npt
import tensorflow as tf
import torch
from scipy.ndimage import gaussian_filter

class NNVis(NNExplainer):
    """Visualise learned features of neural networks via optimisation.""" 
    ## only for torch and tf
    def __init__(self, input_model):
        super().__init__(input_model)
        if isinstance(input_model, torch.nn.Module):
            self.istorch = True
        elif isinstance(input_model, tf.keras.Model):
            self.istorch = False
        else:
            raise TypeError(f"`NNVis` only support Torch or TensorFlow models.")
    
        
    def visualise(
        self,
        layer_identifier: int | str,
        input_shape: tuple[int, ...],
        channel_idx: int,
        lr: float = 1e-3,
        max_iter: int = 200,
        weight_decay: float = 1e-3,
        jitter: int = 2,
        blur_sigma: float = 0.5,
        blur_every: int = 4,
        clipping: tuple[float, float] | None = (0.0, 1.0)
    ) -> npt.NDArray:
        if self.istorch:
            self.objective_fn = lambda x : x[:, channel_idx, :, :].mean()
        else:
            self.objective_fn = lambda x : tf.reduce_mean(x[:, :, :, channel_idx])
        
        X = np.random.normal(0.5, 0.1, size=input_shape)
        
        for step in range(1, max_iter + 1):
            if blur_sigma > 0 and step % blur_every == 0:
                if self.istorch:
                    for i in range(X.shape[0]):
                        for j in range(X.shape[1]):
                            X[i, j] = gaussian_filter(X[i, j], sigma=blur_sigma)
                else:
                    for i in range(X.shape[0]):
                        for j in range(X.shape[-1]):
                            X[i, :, :, j] = gaussian_filter(X[i, :, :, j], sigma=blur_sigma)

            ox, oy = np.random.randint(-jitter, jitter + 1, size=2)
            if self.istorch:
                X_jitter = np.roll(np.roll(X, ox, axis=-2), oy, axis=-1)
            else:
                X_jitter = np.roll(np.roll(X, ox, axis=1), oy, axis=2)
        
            grad = self.model.compute_gradients(
                X_jitter,
                objective_layer=layer_identifier,
                objective_fn=self.objective_fn
            )
            
            if self.istorch:
                grad = np.roll(np.roll(grad, -ox, axis=-2), -oy, axis=-1)
            else:
                grad = np.roll(np.roll(grad, -ox, axis=1), -oy, axis=2)
            
            grad_std = np.std(grad) + 1e-8
            grad = grad / grad_std
            
            X += lr * grad - weight_decay * X
            
            if clipping is not None:
                X = np.clip(X, clipping[0], clipping[1])
            
        return X