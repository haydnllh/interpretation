import torch
import numpy.typing as npt
from typing import Any
from ..model import Model
from ..wrap_model import wrap_model


class LinearModel(Model):
    """
    This is a wrapper for Linear Regression models.
    We support Sklearn LinearRegression, Ridge and Lasso.
    Only neural networks with no hidden layers and activation functions are allowed.
    """
    
    def __init__(self, input_model:torch.nn.Module, device:str=None) -> None:
        """
        Initialise with the input model and a wrapped model for inference
        """
        super().__init__(input_model)
        self.wrapped_model = wrap_model(input_model)
        
    def __call__(self, X:npt.ArrayLike) -> npt.NDArray[Any]:
        return self.wrapped_model(X)
    
    @property
    def coef(self):
        pass
    
    @property
    def intercept(self):
        pass