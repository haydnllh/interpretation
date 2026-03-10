import torch
import tensorflow as tf
import sklearn
import numpy as np
import numpy.typing as npt
import pytest
from interpretation.models.wrapper.wrap_model import wrap_model

def test_pytorch_wrapper():
    pytorch_model = torch.nn.Linear(1, 1)
    model = wrap_model(pytorch_model)
    
    assert model is not None
    assert model.model == pytorch_model
    
    try:
        X = np.zeros((1,1), dtype=np.float32)
        y = model(X)
        
        assert isinstance(y, np.ndarray)
        assert y.shape == (1, 1)
    except Exception as e:
        pytest.fail(f"Pytorch wrapper failed test. Exception: {e}")
        
def test_tf_wrapper():
    tf_model = tf.keras.Sequential(
        [
            tf.keras.Input(shape=(1,)),
            tf.keras.layers.Dense(1)
        ]
    )
    model = wrap_model(tf_model)
    
    assert model is not None
    assert model.model == tf_model
    
    try:
        X = np.zeros((1,1), dtype=np.float32)
        y = model(X)
        
        assert isinstance(y, np.ndarray)
        assert y.shape == (1, 1)
    except Exception as e:
        pytest.fail(f"Tensorflow wrapper failed test. Exception: {e}")
        
def test_sklearn_wrapper():
    sklearn_model = sklearn.linear_model.LinearRegression()
    X_train, y_train = np.zeros((1,1)), np.zeros(1)
    sklearn_model.fit(X_train, y_train)
    
    model = wrap_model(sklearn_model)
    
    assert model is not None
    assert model.model == sklearn_model
    
    try:
        y = model(X_train)
        
        assert isinstance(y, np.ndarray)
        assert y.shape == np.array([1])
    except Exception as e:
        pytest.fail(f"Sklearn wrapper failed test. Exception: {e}")