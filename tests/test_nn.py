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
    
    grad1 = model.compute_gradients(X, objective=1, objective_fn=fn).item()
    grad2 = model.compute_gradients(X, objective=0, objective_fn=fn).item()
    grad3 = model.compute_gradients(X, objective="1", objective_fn=fn).item()
    grad4 = model.compute_gradients(X, objective="0", objective_fn=fn).item()
    grad5 = model.compute_gradients(X, objective=1, objective_fn=fn, wrt=0)
    
    assert grad1 == 8.0
    assert grad2 == 1.5
    assert grad3 == 8.0
    assert grad4 == 1.5
    assert (grad5 == np.array([[2.0, 3.0]])).all()
    
def test_torch_neuron_gradient_numbers():
    torch_model = nn.Sequential(
        nn.Linear(1,2),
        nn.Linear(2,2)
    )
    torch_model[0].weight = nn.Parameter(torch.tensor([[1],[2]], dtype=torch.float32))
    torch_model[1].weight = nn.Parameter(torch.tensor([[2,3],[1,2]], dtype=torch.float32))
    model = wrap_model(torch_model)
    
    X = torch.tensor([[1]], dtype=torch.float32)
    fn = lambda x : x.mean()
    
    grad1 = model.compute_gradients(X, objective=(1,0), objective_fn=fn, wrt=(0,0)).item()
    grad2 = model.compute_gradients(X, objective=(1,1), objective_fn=fn, wrt=(0,0)).item()
    grad3 = model.compute_gradients(X, objective=(1,0), objective_fn=fn, wrt=(0,1)).item()
    grad4 = model.compute_gradients(X, objective=(1,1), objective_fn=fn, wrt=(0,1)).item()
    grad5 = model.compute_gradients(X, objective=(1,0), objective_fn=fn).item()
    grad6 = model.compute_gradients(X, objective=(1,1), objective_fn=fn).item()
    
    assert grad1 == 2.0
    assert grad2 == 1.0
    assert grad3 == 3.0
    assert grad4 == 2.0
    assert grad5 == 8.0
    assert grad6 == 5.0