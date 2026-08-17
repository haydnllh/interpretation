import torch
import numpy.typing as npt
from typing import Any
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
        
    def __call__(self, X:npt.ArrayLike) -> npt.NDArray[Any]:
        "Model inference"
        dtype = next(self.model.parameters()).dtype
        
        with torch.no_grad():
            if isinstance(X, torch.Tensor):
                X_tensor = X.to(self.device, dtype=dtype)
            else:
                X_tensor = torch.tensor(X, device=self.device, dtype=dtype)
            
            output = self.model(X_tensor)
            
        return output.detach().cpu().numpy()
    
    def activation_at_layer(self, X:npt.NDArray, layer_identifier: int | str | list[int | str]) -> list[npt.NDArray]:
        """
        Forward passes input X up to the specified layer.
        layer_identifier can be an integer index or layer name string.
        """ 
        if not isinstance(layer_identifier, list):
            layer_identifier = [layer_identifier]
            
        target_modules = {}
        all_modules = [m for m in self.model.modules() if m != self.model]
            
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