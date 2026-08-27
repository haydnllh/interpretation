import numpy as np
import numpy.typing as npt
import torch
import tensorflow as tf

from ..nn_explainer import NNExplainer
from ....utils.validate_input import validate_input_1d

class SaliencyMap(NNExplainer):
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
        class_idx: int
    ):
        if self.istorch:
            objective_fn = lambda x : x.mean()
        else:
            objective_fn = lambda x : tf.reduce_mean(x)

        grad = self.model.compute_gradients(
            X,
            objective=(-1, class_idx),
            objective_fn=objective_fn
        )
        
        return grad