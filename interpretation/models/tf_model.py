import tensorflow as tf
import keras
import numpy.typing as npt
from typing import Any, Callable
from .model import Model

class TfModel(Model):
    """This is a wrapper for TensorFlow / Keras models"""
    
    def __init__(self, input_model:tf.keras.Model) -> None:
        """Wraps a TensorFlow model to the superclass"""
        
        if not isinstance(input_model, tf.keras.Model):
            raise TypeError("input_model must be an instance of tf.keras.Model")
        
        super().__init__(input_model)
        
        if not self.model.built:
            if hasattr(self.model, "input_shape") and self.model.input_shape:
                self.model.build(self.model.input_shape)
        
        self.dtype = self.model.input_dtype
        
    def __call__(self, X:npt.ArrayLike) -> npt.NDArray[Any]:
        "Model inference"
        
        X_tensor = tf.convert_to_tensor(X, dtype=self.dtype)
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
            
        if isinstance(self.model, keras.Sequential):
            pass
        else:
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
            
            X_tensor = tf.convert_to_tensor(X, dtype=self.dtype)
            activations = intermediate_model(X_tensor, training=False)
            
            if isinstance(activations, list):
                return [a.numpy() for a in activations]
        return activations.numpy()
    
    def compute_gradient(
        self,
        X: npt.NDArray | tf.Tensor,
        objective_layer: int | str,
        objective_fn: Callable[[tf.Tensor], tf.Tensor],
        wrt_layer: int | str | None = None,
    ):
        if isinstance(X, tf.Tensor):
            X_tensor = tf.cast(X, dtype=self.dtype)
        else:
            X_tensor = tf.convert_to_tensor(X, dtype=self.dtype)
            
        is_wrt_input = wrt_layer is None or wrt_layer == "input"
        obj_idx = self._get_layer_idx(objective_layer)
        wrt_idx = None if is_wrt_input else self._get_layer_idx(wrt_layer)
        
        if isinstance(self.model, keras.Sequential):
            with tf.GradientTape() as tape:
                curr = X_tensor
                if is_wrt_input:
                    tape.watch(X_tensor)
                    wrt_act = X_tensor
                else:
                    wrt_act = None
                    
                for i, layer in enumerate(self.model.layers):
                    curr = layer(curr)
                    if not is_wrt_input and i == wrt_idx:
                        wrt_act = curr
                        tape.watch(wrt_act)
                    if i == obj_idx:
                        obj_act = curr
                        break
                    
                loss = objective_fn(obj_act)
            target_tensor = X_tensor if is_wrt_input else wrt_act
            grad = tape.gradient(loss, target_tensor)
            
        else:
            obj_output = self.model.layers[obj_idx].output
        
            if is_wrt_input:
                intermediate_model = tf.keras.Model(
                    inputs=self.model.input,
                    outputs=obj_output
                )
                
                with tf.GradientTape() as tape:
                    tape.watch(X_tensor)
                    obj_act = intermediate_model(X_tensor, training=False)
                    loss = objective_fn(obj_act)
                    
                grad = tape.gradient(loss, X_tensor)
            else:
                wrt_output = self.model.layers[wrt_idx].output
                intermediate_model = tf.keras.Model(
                    inputs=self.model.input,
                    outputs=[wrt_output, obj_output]
                )
                
                with tf.GradientTape() as tape:
                    wrt_act, obj_act = intermediate_model(X_tensor, training=False)
                    tape.watch(wrt_act)
                    loss = objective_fn(obj_act)
                    
                grad = tape.gradient(loss, wrt_act)
                
        if grad is None:
            raise RuntimeError(
                f"Gradients could not be computed for objective layer '{objective_layer}' "
                f"with respect to '{wrt_layer}'."
            )
        
        return grad.numpy()
        
    def _get_layer_output(self, identifier: int | str) -> tf.TypeSpec:
        try:
            if isinstance(identifier, int):
                return self.model.layers[identifier].output
            return self.model.get_layer(name=identifier).output
        except (IndexError, ValueError):
            raise ValueError(
                f"Layer '{identifier}' not found in TensorFlow model."
            )
            
    def _get_layer_idx(self, identifier: int | str) -> int:
        if isinstance(identifier, int):
            if identifier < 0 or identifier >= len(self.model.layers):
                raise ValueError(f"Layer index '{identifier}' out of bounds.")
            return identifier

        for idx, layer in enumerate(self.model.layers):
            if layer.name == identifier:
                return idx
        raise ValueError(f"Layer '{identifier}' not found in TensorFlow model.")