import torch
import numpy.typing as npt
from typing import Any
import sklearn
import torch.nn as nn
import tensorflow as tf
from ..model import Model
from ..wrap_model import wrap_model
from ..pytorch_model import PyTorchModel
from ..sklearn_model import SklearnModel
from ..tf_model import TfModel


class LogisticModel(Model):
    """
    This is a wrapper for Logistic Regression models.
    We support Sklearn LogisticRegression.
    Only neural networks with no hidden layers and sigmoid at the final layer are allowed.
    """
    
    def __init__(self, input_model:torch.nn.Module, device:str=None) -> None:
        """
        Initialise with the input model and a wrapped model for inference
        """
        
        if not isinstance(input_model, (nn.Module, sklearn.base.BaseEstimator, tf.keras.Model)):
            raise TypeError("input_model must be an instance of torch / sklearn / tensorflow.")
        
        if isinstance(input_model, sklearn.base.BaseEstimator) and \
            not isinstance(input_model, sklearn.linear_model.LogisticRegression):
                raise TypeError("Unsupported scikit-learn model. Only LinearRegression, Ridge, and Lasso are supported.")
            
        if isinstance(input_model, nn.Module) and not \
            (
                isinstance(input_model, nn.Sequential)
                and len(input_model) == 2
                and isinstance(input_model[0], nn.Linear)
                and isinstance(input_model[1], nn.Sigmoid)
            ):
                raise TypeError("Pytorch model must only contain one nn.Linear layer and one Sigmoid activation function layer.")
            
        if isinstance(input_model, tf.keras.Model) and not\
            (
                isinstance(input_model, tf.keras.Sequential)
                and len(input_model.layers) == 1
                and isinstance(input_model.layers[0], tf.keras.layers.Dense)
                and input_model.layers[0].activation == tf.keras.activations.sigmoid
            ):
                raise TypeError("TensorFlow model must be an tf.keras.Sequential containing exactly one Dense layer with a sigmoid activation.")
                    
        super().__init__(input_model)
        self.wrapped_model = wrap_model(input_model)
        
    def __call__(self, X:npt.ArrayLike) -> npt.NDArray[Any]:
        return self.wrapped_model(X)
    
    @property
    def coef(self):
        match self.wrapped_model:
            case SklearnModel():
                return self.model.coef_.reshape(-1)
            case PyTorchModel():
                return self.model.weight.detach().squeeze().numpy()
            case TfModel():
                return self.model.layers[0].get_weights()[0].reshape(-1)
    
    @property
    def intercept(self):
        match self.wrapped_model:
            case SklearnModel():
                return self.model.intercept_.reshape(-1)
            case PyTorchModel():
                return self.model.bias.detach().squeeze().numpy()
            case TfModel():
                return self.model.layers[0].get_weights()[1].reshape(-1)