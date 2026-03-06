from typing import Any
import numpy.typing as npt

class Model:
    """This is a superclass for all models"""
    
    def __init__(self, input_model: Any) -> None:
        """Wrapping a model as a generic model"""
        if isinstance(input_model, Model):
            self.model = input_model.model
        else:
            self.model = input_model
            
    def __call__(self, X:npt.ArrayLike) -> npt.NDArray[Any]:
        """
        This is meant to be overridden by subclasses
        
        Returns the model's prediction. Similar to .predict from Scikit-Learn or __call__ from PyTorch
        """
        raise NotImplementedError()