from typing import Any
import numpy.typing as npt
from abc import ABC, abstractmethod

class Model(ABC):
    """This is a superclass for all models"""
    
    def __init__(self, input_model: Any) -> None:
        """Wrapping a model as a generic model"""
        if isinstance(input_model, Model):
            self.model = input_model.model
        else:
            self.model = input_model
            
    @abstractmethod
    def __call__(self, X:npt.ArrayLike) -> npt.NDArray[Any]:
        """
        Must be implemented by subclasses.
        
        Returns the model's prediction. Similar to .predict from Scikit-Learn or __call__ from PyTorch
        """
        pass