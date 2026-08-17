from typing import Any
import numpy.typing as npt
import sklearn
from sklearn.neural_network import MLPClassifier, MLPRegressor
import torch
import torch.nn as nn
import tensorflow as tf

from .model import Model
from .wrap_model import wrap_model
from .pytorch_model import PyTorchModel
from .sklearn_model import SklearnModel
from .tf_model import TfModel

class NNModel(Model):
    """
    This is a wrapper for Neural Networks.
    We support support any Pytorch and Tensorflow neural networks.
    """
    def __init__(self, input_model:Any) -> None:
        """
        Initialise with the input model and a wrapped model for inference.
        """
        if not isinstance(input_model, (nn.Module, tf.keras.Model)):
            raise TypeError("input_model must be an instance of Torch or Tensorflow.")
        
        super().__init__(input_model)
        self.wrapped_model = wrap_model(input_model)
        
    def __call__(self, X: npt.ArrayLike) -> npt.NDArray[Any]:
        return self.wrapped_model(X)
    
    @property
    def weights(self) -> list[npt.NDArray[Any]]:
        """Extracts the weight matrices for each trainable layer across backends."""
        match self.wrapped_model:
            case SklearnModel():
                return self.model.coefs_
            case PyTorchModel():
                return [
                    param.detach().cpu().numpy()
                    for name, param in self.model.named_parameters()
                    if "weight" in name
                ]
            case TfModel():
                return [
                    layer.get_weights()[0]
                    for layer in self.model.layers
                    if len(layer.get_weights()) > 0
                ]

    @property
    def biases(self) -> list[npt.NDArray[Any]]:
        """Extracts the bias vectors for each trainable layer across backends."""
        match self.wrapped_model:
            case SklearnModel():
                return self.model.intercepts_
            case PyTorchModel():
                return [
                    param.detach().cpu().numpy()
                    for name, param in self.model.named_parameters()
                    if "bias" in name
                ]
            case TfModel():
                return [
                    layer.get_weights()[1]
                    for layer in self.model.layers
                    if len(layer.get_weights()) > 1
                ]
                
    def activation_at_layer(self, X:npt.NDArray, layer_identifier: int | str | list[int | str]) -> list[npt.NDArray]:
        return self.wrapped_model.activation_at_layer(X, layer_identifier)
    