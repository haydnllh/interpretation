import torch
import numpy.typing as npt
from typing import Any
from ..model import Model
from ..wrap_model import wrap_model


class LogisticModel(Model):
    """
    This is a wrapper for Logistic Regression models.
    We support Sklearn LinearRegression.
    Only neural networks with no hidden layers and sigmoid at the final layer are allowed.
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