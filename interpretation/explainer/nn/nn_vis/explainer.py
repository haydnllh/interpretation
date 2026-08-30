import numpy as np
import numpy.typing as npt
import tensorflow as tf
import torch
from scipy.ndimage import gaussian_filter

from ..nn_explainer import NNExplainer
from ....utils.validate_input import validate_input_numpy

class NNVis(NNExplainer):
    """Visualise learned features of neural networks via gradient ascent. This is ideal for visualisation of models that use visual features, e.g. image classification""" 
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
        channel_idx: int = 0,
        lr: float = 1e-3,
        max_iter: int = 200,
        weight_decay: float = 1e-3,
        max_jitter: int = 2,
        blur_sigma: float = 0.5,
        blur_every: int = 4,
        clipping: tuple[float, float] | None = None
    ) -> npt.NDArray:
        """Generate the feature visualisation, based on optimisation of maximum activation with respect to a given layer by gradient ascent.
        We use L2, Gaussian blurring and jittering regularisation strategies to prevent noisy outputs.

        Parameters
        ----------
        layer_identifier : int | str
            The layer we maximise the input with respect to.
        input_shape : tuple[int, ...]
            The shape of the input we are optimising.
        channel_idx : int
            The channel of the image we are maximising on, by default 0.
        lr : float, optional
            Learning rate of the gradient ascent, by default 1e-3
        max_iter : int, optional
            Maximum number of iterations of the gradient ascent by default 200
        weight_decay : float, optional
            L2 regularisation strength of the gradient ascent, by default 1e-3
        max_jitter : int, optional
            Maximum number of pixels jitter, by default 2
        blur_sigma : float, optional
            Sigma of the Gaussian blurring, by default 0.5
        blur_every : int, optional
            Number of iterations between each Gaussian blurring, by default 4
        clipping : tuple[float, float] | None, optional
            Clipping of the image values to prevent overflow, by default (0.0, 1.0)

        Returns
        -------
        npt.NDArray
            The input generated that maximises the activation at the specified layer.

        Raises
        ------
        ValueError
            Input shape is not 1-d, 2-d or 3-d.
        """
        X = validate_input_numpy(X)
        
        if self.istorch:
            objective_fn = lambda x : x[:, channel_idx, :, :].mean()
        else:
            objective_fn = lambda x : tf.reduce_mean(x[:, :, :, channel_idx])
            
        if self.istorch and input_shape.ndim == 3:
            objective_fn = lambda x : x[channel_idx, :, :].mean()
        elif not self.istorch and input_shape.ndim == 3:
            objective_fn = lambda x : tf.reduce_mean(x[:, :, channel_idx])
        elif self.istorch and (input_shape.ndim == 2 or input_shape == 1):
            objective_fn = lambda x : x.mean()
        elif not self.istorch and (input_shape.ndim == 2 or input_shape == 1):
            objective_fn = lambda x : tf.reduce_mean(x)
        else:
            raise ValueError("Input shape can only be 1-d, 2-d or 3-d.")

        X = np.random.normal(0.5, 0.1, size=input_shape)[None, :]
        
        for step in range(1, max_iter + 1):
            if blur_sigma > 0 and step % blur_every == 0:
                if input_shape.ndim < 3:
                    X = gaussian_filter(X, sigma=blur_sigma)
                else:
                    if self.istorch:
                        for j in range(X.shape[1]):
                            X[0, j] = gaussian_filter(X[0, j], sigma=blur_sigma)
                    else:
                        for j in range(X.shape[-1]):
                            X[0, :, :, j] = gaussian_filter(X[0, :, :, j], sigma=blur_sigma)

            ox, oy = np.random.randint(-max_jitter, max_jitter + 1, size=2)
            
            if input_shape.ndim == 1:
                X_jitter = np.roll(X, ox)
            elif input_shape.ndim == 2:
                X_jitter = np.roll(np.roll(X, ox, axis=0), oy, axis=1)
            else:
                if self.istorch:
                    X_jitter = np.roll(np.roll(X, ox, axis=-2), oy, axis=-1)
                else:
                    X_jitter = np.roll(np.roll(X, ox, axis=1), oy, axis=2)
        
            grad = self.model.compute_gradients(
                X_jitter,
                objective=layer_identifier,
                objective_fn=objective_fn
            )
            
            if input_shape.ndim == 1:
                grad = np.roll(grad, -ox)
            elif input_shape.ndim == 2:
                grad = np.roll(np.roll(grad, -ox, axis=0), -oy, axis=1)
            else:
                if self.istorch:
                    grad = np.roll(np.roll(grad, -ox, axis=-2), -oy, axis=-1)
                else:
                    grad = np.roll(np.roll(grad, -ox, axis=1), -oy, axis=2)
            
            grad_std = np.std(grad) + 1e-8
            grad = grad / grad_std
            
            X += lr * grad - weight_decay * (X ** 2)
            
            if clipping is not None:
                X = np.clip(X, clipping[0], clipping[1])
            
        return X