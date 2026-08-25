import torch
import torch.nn as nn
import tensorflow as tf
import sklearn
import numpy as np
import pytest
from interpretation.models import wrap_model
        

def test_torch_gradient_numbers():
    torch_model = nn.Sequential(
        nn.Linear(1,2),
        nn.Linear(2,1)
    )
    torch_model[0].weight = nn.Parameter(torch.tensor([[1],[2]], dtype=torch.float32))
    torch_model[1].weight = nn.Parameter(torch.tensor([[2,3]], dtype=torch.float32)) 
    model = wrap_model(torch_model)
    
    X = torch.tensor([[1]], dtype=torch.float32)
    fn = lambda x : x.mean()
    
    grad1 = model.compute_gradients(X, objective_layer=1, objective_fn=fn).item()
    grad2 = model.compute_gradients(X, objective_layer=0, objective_fn=fn).item()
    grad3 = model.compute_gradients(X, objective_layer="1", objective_fn=fn).item()
    grad4 = model.compute_gradients(X, objective_layer="0", objective_fn=fn).item()
    grad5 = model.compute_gradients(X, objective_layer=1, objective_fn=fn, wrt_layer=0)
    
    assert grad1 == 8.0
    assert grad2 == 1.5
    assert grad3 == 8.0
    assert grad4 == 1.5
    assert (grad5 == np.array([[2.0, 3.0]])).all()