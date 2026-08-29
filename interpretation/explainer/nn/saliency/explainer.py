import numpy as np
import numpy.typing as npt
import torch
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.axis import Axis
from matplotlib.figure import Figure

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
            Instance to produce the saliency map on
        class_idx : int
            The objective class of the saliency map
        method : str, optional
            Algorithm used for computing the saliency map, by default "SmoothGrad"
        n_samples : int, optional
            Number of samples to average from when using "SmoothGrad", by default 50

        Returns
        -------
        npt.NDArray
            Resulting saliency map, the partial derivative of class score w.r.t. input, shape is the same as the input
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
            
        elif method == "SmoothGrad":
            grad = smooth_grad(
                self.model,
                X,
                class_idx=class_idx,
                objective_fn=objective_fn,
                n_samples=n_samples
            )
            
        else:
            raise ValueError(f"Method {method} is not supported, only valid methods are 'vanilla' and 'SmoothGrad'.")
        
        return grad
    
    def plot_map(
        self,
        X: npt.NDArray,
        class_idx: int,
        method: str = "SmoothGrad",
        n_samples: int = 50,
        ax: Axis | None = None
    ):
        """Plots the saliency map of the input on an axis.

        Parameters
        ----------
        X : npt.NDArray
            Instance to produce the saliency map on
        class_idx : int
            The objective class of the saliency map
        method : str, optional
            Algorithm used for computing the saliency map, by default "SmoothGrad"
        n_samples : int, optional
            Number of samples to average from when using "SmoothGrad", by default 50
        ax : Axis | None, optional
            Matplotlib axis to plot on, by default None

        Returns
        -------
        fig : Figure
            Matplotlib Figure object containing the saliency map axis
        ax : Axis
            Matplotlib Axis object that the saliency map is plotted on
        """
        map = self.compute_map(X, class_idx, method, n_samples)
        
        map = np.transpose(map.squeeze(), (1,2,0))
        map = np.max(np.abs(map), axis=-1)
        map = (map - np.min(map)) / (np.max(map) - np.min(map) + 1e-8)
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 8))
        else:
            fig = ax.figure
            
        img = ax.imshow(map, cmap="jet")
        ax.axis("off")
        
        fig.colorbar(img, ax=ax)
        
        return fig, ax