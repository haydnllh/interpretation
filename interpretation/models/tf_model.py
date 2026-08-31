import tensorflow as tf
import keras
import numpy.typing as npt
from typing import Any, Callable, Tuple
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
        layer_identifier = self._get_layer_idx(layer_identifier)
        X_tensor = tf.convert_to_tensor(X, dtype=self.dtype)
        
        if not isinstance(layer_identifier, list):
            layer_identifier = [layer_identifier]
            
        if isinstance(self.model, keras.Sequential):
            activation_map = {}
            activations = []
            curr = X_tensor
            
            for i, layer in enumerate(self.model.layers):
                curr = layer(curr)
                activation_map[i] = curr
                
            for layer_idx in layer_identifier:
                activations.append(activation_map[layer_idx].numpy())
                
        else:
            intermediate_model = tf.keras.Model(
                inputs=self.model.input,
                outputs=layer_identifier
            )
            
            activations = intermediate_model(X_tensor, training=False)
            
            if isinstance(activations, list):
                activations =  [a.numpy() for a in activations]
            
        return activations
    
    def compute_gradients(
        self,
        X: npt.NDArray | tf.Tensor,
        objective: int | str | Tuple[int, int],
        objective_fn: Callable[[tf.Tensor], tf.Tensor],
        wrt: int | str | Tuple[int, int] |None = None,
    ):
        r"""Computes the gradient \(\frac{\partial \text{Objective}}{\partial X}\).

        `objective` is Objective, `wrt` is X.

        The gradient is calculated after applying the objective function to the objective layer w.r.t the wrt layer.
        
        Tuple forms of `objective` or `wrt` specifies specific neurons, e.g. (0, 1) means the second neuron of the first layer.

        Note: When no `wrt` is provided, it will default to `'input'`.
        """
        if isinstance(X, tf.Tensor):
            X_tensor = tf.cast(X, dtype=self.dtype)
        else:
            X_tensor = tf.convert_to_tensor(X, dtype=self.dtype)
            
        is_wrt_input = wrt is None or wrt == "input"
        obj_idx = self._get_layer_idx(objective)
        wrt_idx = None if is_wrt_input else self._get_layer_idx(wrt)
        
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
                    
                if isinstance(objective, Tuple):
                    loss = objective_fn(tf.squeeze(obj_act)[objective[1]])
                else:
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
                    
                    if isinstance(objective, Tuple):
                        loss = objective_fn(tf.squeeze(obj_act)[objective[1]])
                    else:
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
                    
                    if isinstance(objective, Tuple):
                        loss = objective_fn(tf.squeeze(obj_act)[objective[1]])
                    else:
                        loss = objective_fn(obj_act)
                    
                grad = tape.gradient(loss, wrt_act)
                
        if grad is None:
            raise RuntimeError(
                f"Gradients could not be computed for objective layer '{objective}' "
                f"with respect to '{wrt}'."
            )
        
        if isinstance(wrt, Tuple):
            grad = tf.squeeze(grad)[wrt[1]]

        return grad.numpy()
            
    def _get_layer_idx(self, identifier: int | str | list[int | str] | Tuple[int, int]) -> int:
        if isinstance(identifier, int):
            if identifier < 0 or identifier >= len(self.model.layers):
                raise ValueError(f"Layer index '{identifier}' out of bounds.")
            return identifier
        
        if isinstance(identifier, Tuple):
            return identifier[0]
        
        if isinstance(identifier, list):
            layer_map = {}
            layer_indices = []
            
            for idx, layer in enumerate(self.model.layers):
                layer_map[layer] = idx
            
            for layer in identifier:
                if isinstance(layer, int):
                    layer_indices.append(layer)
                elif isinstance(layer, str):
                    layer_indices.append(layer_map[layer])
                else:
                    raise TypeError("Layer identifiers can only be int or str.")
            
            return layer_indices

        for idx, layer in enumerate(self.model.layers):
            if layer.name == identifier:
                return idx
        raise ValueError(f"Layer '{identifier}' not found in TensorFlow model.")