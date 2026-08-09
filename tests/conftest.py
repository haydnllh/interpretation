import matplotlib
matplotlib.use("Agg")

import pytest
from sklearn.datasets import make_regression, make_classification
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier


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