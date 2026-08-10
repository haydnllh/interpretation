import numpy as np

def validate_input_label(X, y):
    if not isinstance(X, np.ndarray):
        raise TypeError("Input must be an instance of np.ndarray")
    
    if X.ndim != 2:
        raise ValueError("Input expected to be a 2-d array")
    
    if not isinstance(y, np.ndarray):
        raise TypeError("Target must be an instance of np.ndarray")
    
    if y.ndim not in (1, 2):
        raise ValueError("Target expected to be a 1-d or 2-d array")
    
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of rows.")
    
def validate_input_1d(X):
    if not isinstance(X, np.ndarray):
        raise TypeError("Input must be an instance of np.ndarray")
    
    if X.ndim > 1 and not all(s == 1 for s in X.shape[:-1]):
        raise ValueError("Input expected to be a 1-d vector")
    
def validate_input_2d(X):
    if not isinstance(X, np.ndarray):
        raise TypeError("Input must be an instance of np.ndarray")
    
    if X.ndim != 2:
        raise ValueError("Input expected to be a 2-d array")