import sklearn
import torch.nn as nn
import tensorflow as tf
import numpy.typing as npt
from typing import Any
from .model import Model
from .wrap_model import wrap_model
from .pytorch_model import PyTorchModel
from .sklearn_model import SklearnModel
from .tf_model import TfModel


class LinearModel(Model):
    """
    This is a wrapper for Linear Regression models.
    We support Sklearn LinearRegression, Ridge and Lasso.
    Only neural networks with one layer and no activation functions are allowed.
    """
    
    def __init__(self, input_model) -> None:
        """
        Initialise with the input model and a wrapped model for inference
        """
        
        if not isinstance(input_model, (nn.Module, sklearn.base.BaseEstimator, tf.keras.Model)):
            raise TypeError("input_model must be an instance of torch / sklearn / tensorflow.")
        
        if isinstance(input_model, sklearn.base.BaseEstimator) and \
            not isinstance(input_model, (sklearn.linear_model.LinearRegression, sklearn.linear_model.Ridge, sklearn.linear_model.Lasso)):
                raise TypeError("Unsupported scikit-learn model. Only LinearRegression, Ridge, and Lasso are supported.")
            
        if isinstance(input_model, nn.Module) and \
            (
                (not isinstance(input_model, nn.Linear) and  \
                not (
                    isinstance(input_model, nn.Sequential) 
                    and len(input_model) == 1 
                    and isinstance(input_model[0], nn.Linear)
                    )
                )
            ):
                raise TypeError("Pytorch model must only contain one nn.Linear layer with no activation function.")
            
        if isinstance(input_model, tf.keras.Model) and not\
            (
                isinstance(input_model, tf.keras.Sequential)
                and len(input_model.layers) == 1
                and isinstance(input_model.layers[0], tf.keras.layers.Dense)
                and input_model.layers[0].activation == tf.keras.activations.linear
            ):
                raise TypeError("TensorFlow model must be an tf.keras.Sequential containing exactly one Dense layer with a linear activation.")
            
            
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