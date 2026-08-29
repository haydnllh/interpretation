import numpy as np
import numpy.typing as npt
import torch
import tensorflow as tf

from ..nn_explainer import NNExplainer
from ....utils.validate_input import validate_input_1d
from .smooth_grad import smooth_grad

class SaliencyMap(NNExplainer):
    r""" 
    Saliency is a map of the derivatives of the class score with respect to the input pixels / features:
    
    \( \frac{\partial S_c}{\partial x} \)
    
    where \( S_c = \text{class score}\) and \(x = \text(input)\)
    
    It shows the sensitivity of each pixel / feature, higher values can mean higher importance.
    """
    def __init__(self, input_model):
        super().__init__(input_model)
        if isinstance(input_model, torch.nn.Module):
            self.istorch = True
        elif isinstance(input_model, tf.keras.Model):
            self.istorch = False
        else:
            raise TypeError(f"`SaliencyMap` only support Torch or TensorFlow models.")
        
    def compute_map(
        self,
        X: npt.NDArray,
        class_idx: int,
        method: str = "SmoothGrad",
        n_samples: int = 50
    ) -> npt.NDArray:
        """Computes the saliency map of the input

        Parameters
        ----------
        X : npt.NDArray
            Instance to produce the saliency map on.
        class_idx : int
            The objective class of the saliency map.
        method : str, optional
            Algorithm used for computing the saliency map, by default "vanilla"

        Returns
        -------
        _type_
            _description_
        """
        if self.istorch:
            objective_fn = lambda x : x.mean()
        else:
            objective_fn = lambda x : tf.reduce_mean(x)

        if method == "vanilla":
            grad = self.model.compute_gradients(
                X,
                objective=(-1, class_idx),
                objective_fn=objective_fn
            )
            
            return grad
        elif method == "SmoothGrad":
            grad = smooth_grad(
                self.model,
                X,
                class_idx=class_idx,
                objective_fn=objective_fn,
                n_samples=n_samples
            )
            
            return grad
        else:
            raise ValueError(f"Method {method} is not supported, only valid methods are 'vanilla' and 'SmoothGrad'.")