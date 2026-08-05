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
                X = X.to(self.device, dtype=dtype)
            else:
                X = torch.tensor(X, device=self.device, dtype=dtype)
            
            output = self.model(X)
            
        return output.detach().cpu().numpy()