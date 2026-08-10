import matplotlib
matplotlib.use("Agg")

import pytest
from sklearn.datasets import make_regression, make_classification
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import torch.nn as nn
import keras
from keras import layers


@pytest.fixture(scope="session")
def regressor():
    X, y = make_regression(
        n_samples=200,
        n_features=4,
        noise=0.1,
        random_state=42
    )

    model = LinearRegression()
    model.fit(X, y)

    return model, X, y


@pytest.fixture(scope="session")
def binary_classifier():
    X, y = make_classification(
        n_samples=200,
        n_features=4,
        n_informative=3,
        n_redundant=0,
        n_classes=2,
        random_state=42
    )

    model = LogisticRegression()
    model.fit(X, y)

    return model, X, y


@pytest.fixture(scope="session")
def multiclass_classifier():
    X, y = make_classification(
        n_samples=300,
        n_features=4,
        n_informative=4,
        n_redundant=0,
        n_classes=3,
        n_clusters_per_class=1,
        random_state=42
    )

    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)

    return model, X, y

def torch_classifier(n_input, n_output):
    class NNClassifier(nn.Module):
        def __init__(self, n_input, n_output):
            super().__init__()

            self.network = nn.Sequential(
                nn.Linear(n_input, n_input * 2),
                nn.ReLU(),
                nn.Linear(n_input * 2, n_output),
                nn.Softmax(dim=1)
            )

        def forward(self, x):
            return self.network(x)
    
    return NNClassifier(n_input, n_output)

def torch_regressor(n_input, n_output):
    class NNRegressor(nn.Module):
        def __init__(self, n_input, n_output):
            super().__init__()

            self.network = nn.Sequential(
                nn.Linear(n_input, n_input * 2),
                nn.ReLU(),
                nn.Linear(n_input * 2, n_output),
            )

        def forward(self, x):
            return self.network(x)
        
    return NNRegressor(n_input, n_output)


    
def tf_classifier(n_input, n_output):
    model = keras.Sequential(
        [
            layers.Input(shape=(n_input,)),
            layers.Dense(n_input * 2, activation="relu"),
            layers.Dense(n_output, activation="softmax")
        ]
    )
    return model
    
def tf_regressor(n_input, n_output):
    model = keras.Sequential(
        [
            layers.Input(shape=(n_input,)),
            layers.Dense(n_input * 2, activation="relu"),
            layers.Dense(n_output)
        ]
    )
    return model