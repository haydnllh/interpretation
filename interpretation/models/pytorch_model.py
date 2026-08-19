import torch
import numpy.typing as npt
from typing import Any
from typing import Callable, Tuple, Any, Dict
from .model import Model

class PyTorchModel(Model):
    """This is a wrapper for PyTorch models"""
    
    def __init__(
        self, 
        input_model: torch.nn.Module, 
        device: torch.device | str = None
    ) -> None:
        """Wraps a PyTorch model to the superclass"""
        
        if not isinstance(input_model, torch.nn.Module):
            raise TypeError("input_model must be an instance of torch.nn.Module")
        
        super().__init__(input_model)
        
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model.to(self.device)
        self.model.eval()
        
        self.layer_dict = {
            name: module
            for name, module in self.model.named_modules()
            if len(list(module.children())) == 0
        }
        
    def __call__(self, X:npt.NDArray | torch.Tensor) -> npt.NDArray[Any]:
        "Model inference"
        dtype = next(self.model.parameters()).dtype
        
        with torch.no_grad():
            if isinstance(X, torch.Tensor):
                X_tensor = X.to(self.device, dtype=dtype)
            else:
                X_tensor = torch.tensor(X, device=self.device, dtype=dtype)
            
            output = self.model(X_tensor)
            
        return output.detach().cpu().numpy()
    
    def activation_at_layer(
        self, 
        X: npt.NDArray | torch.Tensor, 
        layer_identifier: int | str | list[int | str]
    ) -> list[npt.NDArray]:
        """
        Forward passes input X up to the specified layer.
        layer_identifier can be an integer index or layer name string.
        """ 
        if not isinstance(layer_identifier, list):
            layer_identifier = [layer_identifier]
            
        target_modules = {}
        all_modules = [m for m in self.model.modules()][1:]
            
        for out_idx, identifier in enumerate(layer_identifier):
            if isinstance(identifier, int):
                target_module = all_modules[identifier]
            elif isinstance(identifier, str):
                target_module = dict(self.model.named_modules()).get(identifier)
                
            if target_module is None:
                raise ValueError(f"Layers '{layer_identifier}' not found in PyTorch model.")
            
            target_modules[target_module] = out_idx
        
        activations = [None] * len(layer_identifier)
        handles = []
        
        def hook_generator(out_idx):
            def hook(module, input, output):
                activations[out_idx] = output.detach().cpu().numpy()
            return hook
            
        for module, out_idx in target_modules.items():
            handles.append(module.register_forward_hook(hook_generator(out_idx))) #runs the hook code when forward passed to the layer
        
        try:
            self.__call__(X)
        finally:
            for h in handles:
                h.remove()
        
        return activations
    
    def compute_gradients(
        self,
        X: npt.NDArray | torch.Tensor,
        objective_layer: int | str,
        objective_fn: Callable[[torch.Tensor], torch.tensor],
        wrt_layer: int | str = None
    ) -> npt.NDArray:
        r"""Computes the gradient \(\frac{\partial \text{Objective}}{\partial X}\).

        `objective_layer` is Objective, `wrt_layer` is X.

        The gradient is calculated after applying the objective function to the objective layer w.r.t the wrt layer.

        Note: When no `wrt_layer` is provided, it will default to `'input'`.
        """
        dtype = next(self.model.parameters()).dtype
        
        if isinstance(X, torch.Tensor):
            X_tensor = X.to(self.device, dtype=dtype)
        else:
            X_tensor = torch.tensor(X, device=self.device, dtype=dtype)        
                    
        activations = {}
        hooks = []
        
        def hook_generator(name):
            def hook(module, input, output):
                activations[name] = output
            return hook
            
        obj_name = self._layer_int_to_name(objective_layer)
        wrt_name = self._layer_int_to_name(wrt_layer) if wrt_layer is not None else "input"
        
        hooks.append(self.layer_dict[obj_name].register_forward_hook(hook_generator("obj")))
        if wrt_name != "input":
            hooks.append(self.layer_dict[wrt_name].register_forward_hook(hook_generator("wrt")))
            
        try:
            if wrt_name == "input":
                target_wrt = X_tensor
                target_wrt.requires_grad_()
            
                _ = self.model(target_wrt)
            else:
                def retain_grad_hook(module, input, output):
                    output.retain_grad()
                    activations["wrt"] = output
                    
                hooks.append(self.layer_dict[wrt_name].register_forward_hook(retain_grad_hook))
                
                _ = self.model(X_tensor)
                target_wrt = activations["wrt"]
                
            obj_activation = activations["obj"]
            loss = objective_fn(obj_activation)
            
            if target_wrt.grad is not None:
                target_wrt.grad.zero_()
            loss.backward()
            
            grad = target_wrt.grad.detach().cpu().numpy()

            return grad
        
        finally:
            for h in hooks:
                h.remove()
        
    def _layer_int_to_name(self, layer_id: int | str) -> str:
        if isinstance(layer_id, int):
            return list(self.layer_dict.keys())[layer_id]
        return layer_id