from sklearn.base import BaseEstimator, is_classifier
import numpy as np
import numpy.typing as npt 
from typing import Any
from .model import Model

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
        
        if is_classifier(self.model):
            pred = self.model.predict_proba(X)
            return pred[:, 1] if pred.shape[-1] == 2 else pred
            
        return self.model.predict(X)