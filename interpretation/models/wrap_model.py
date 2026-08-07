from .pytorch_model import PyTorchModel
from .sklearn_model import SklearnModel
from .tf_model import TfModel

def wrap_model(input_model):
    "This function wraps models automatically to the Model class"
    import torch
    import tensorflow as tf
    import sklearn
    
    if isinstance(input_model, torch.nn.Module):
        return PyTorchModel(input_model)
    elif isinstance(input_model, sklearn.base.BaseEstimator):
        return SklearnModel(input_model)
    elif isinstance(input_model, tf.keras.Model):
        return TfModel(input_model)
    else:
        raise TypeError("Model type unsupported. Model can only be PyTorch, TensorFlow or Scikit-Learn.")