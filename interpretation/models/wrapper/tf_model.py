import tensorflow as tf
import numpy.typing as npt
from typing import Any
from .model import Model

class TfModel(Model):
    """This is a wrapper for TensorFlow / Keras models"""
    
    def __init__(self, input_model:tf.keras.Model) -> None:
        """Wraps a TensorFlow model to the superclass"""
        
        if not isinstance(input_model, tf.keras.Model):
            raise TypeError("input_model must be an instance of tf.keras.Model")
        
        super().__init__(input_model)
        
    def __call__(self, X:npt.ArrayLike) -> npt.NDArray[Any]:
        "Model inference"
        
        X = tf.convert_to_tensor(X)
        output = self.model(X, training=False)
            
        return output.numpy()