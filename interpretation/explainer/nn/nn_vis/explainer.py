from ..nn_explainer import NNExplainer
import numpy as np
import numpy.typing as npt
import tensorflow as tf
import torch

class NNVis(NNExplainer):
    """Visualise learned features of neural networks via optimisation.""" 
    ## only for torch and tf
    def __init__(self, input_model):
        super().__init__(input_model)
        
        if isinstance(input_model, torch.nn.Module):
            self.objective_fn = lambda x : x[:, 0, :, :].mean()
        elif isinstance(input_model, tf.keras.Model):
            self.objective_fn = lambda x : tf.reduce_mean(x)
        else:
            raise TypeError(f"`NNVis` only support Torch or TensorFlow models.")
    
    def visualise(
        self,
        layer_identifier: int | str,
        input_shape: tuple[int, ...],
        lr: float = 1e-3,
        threshold: float = 1e-3,
        max_iter: int = 1000,
        clipping: tuple[float, float] | None = None
    ) -> npt.NDArray:
        X = np.random.normal(0, 1, size=input_shape)
        
        for _ in range(max_iter):
            grad = self.model.compute_gradients(
                X,
                objective_layer=layer_identifier,
                objective_fn=self.objective_fn
            )
            
            if np.abs(grad).mean() < threshold:
                break
            X += lr * grad
            
            if clipping is not None:
                X = np.clip(X, clipping[0], clipping[1])
            
        return X