from sklearn.base import BaseEstimator
import numpy as np
import numpy.typing as npt 
from typing import Any
from wrapper.model import Model

class SklearnModel(Model):
    """This is a wrapper for Sklearn models"""
    
    def __init__(self, input_model:BaseEstimator) -> None:
        """Wraps a PyTorch model to the superclass"""
        
        if not isinstance(input_model, BaseEstimator):
            raise TypeError("input_model must be an instance of sklearn.base.BaseEstimator")
        
        super().__init__(input_model)
        
    def __call__(self, X:npt.ArrayLike) -> npt.NDArray[Any]:
        "Model inference"
        
        X = np.asarray(X)
        output = self.model.predict(X)
        
        return output