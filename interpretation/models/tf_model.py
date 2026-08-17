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
        
        X_tensor = tf.convert_to_tensor(X, dtype=self.model.input_dtype)
        output = self.model(X_tensor, training=False)
            
        return output.numpy()
    
    def activation_at_layer(self, X:npt.NDArray, layer_identifier: int | str | list[int | str]) -> list[npt.NDArray]:
        """
        Forward passes input X up to the specified layer.
        layer_identifier can be an integer index or layer name string.
        """ 
        target_layers = []
        
        if not isinstance(layer_identifier, list):
            layer_identifier = [layer_identifier]
        
        for identifier in layer_identifier:
            try:
                if isinstance(identifier, int):
                    target_layers.append(self.model.layers[identifier].output)
                else:
                    target_layers.append(self.model.get_layer(name=identifier).output)
            except (IndexError, ValueError):
                raise ValueError(f"Layer '{identifier}' not found in TensorFlow model.")
            
        if not target_layers:
            raise ValueError(f"Layers '{layer_identifier}' not found in Tensorflow model.")
        
        intermediate_model = tf.keras.Model(
            inputs=self.model.input,
            outputs=target_layers
        )
        
        X_tensor = tf.convert_to_tensor(X, dtype=self.model.input_dtype)
        activations = intermediate_model(X_tensor, training=False)
        
        if isinstance(activations, list):
            return [a.numpy() for a in activations]
        return activations.numpy()