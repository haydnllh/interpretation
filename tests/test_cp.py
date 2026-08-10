# test multi-outputs
# test regressor and classifier

import numpy as np
from interpretation.explainer.agnostic import CPExplainer
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import pytest

def test_regressor(regressor):
    model, X, y = regressor
    
    explainer = CPExplainer(model, X)
    result = explainer.explain(
        X[0],
        feature_idx=0,
        n_grid=50
    )
    
    assert result is not None
    assert result.shape == (50,)
    
def test_binary_classifier(binary_classifier):
    model, X, y = binary_classifier
    
    explainer = CPExplainer(model, X)
    result = explainer.explain(
        X[0],
        feature_idx=0,
        n_grid=50
    )
    
    assert result is not None
    assert result.shape == (50,)
    
def test_multiclass_classifier(multiclass_classifier):
    model, X, y = multiclass_classifier
    
    explainer = CPExplainer(model, X)
    result = explainer.explain(
        X[0],
        feature_idx=0,
        n_grid=50
    )
    
    assert result is not None
    assert result.shape == (50, len(np.unique(y)))
    
def test_plots(regressor):
    model, X, y = regressor
    
    explainer = CPExplainer(model, X)
    
    ax = explainer.plot(
        X[0],
        feature_idx=0,
    )
    
    assert ax is not None
    assert isinstance(ax, Axes)

    plt.close('all')

def test_plots_multiclass(multiclass_classifier):
    model, X, y = multiclass_classifier
    
    explainer = CPExplainer(model, X)
    
    ax = explainer.plot(
        X[0],
        feature_idx=0,
        output_idx=0
    )
    
    assert ax is not None
    assert isinstance(ax, Axes)

    plt.close('all')
    
def test_input(regressor):
    model, X, y = regressor
    
    explainer = CPExplainer(model, X)
    
    result = explainer.explain(
        X[0][None, None, :], 
        feature_idx=0,
        n_grid=50
    )
    assert result is not None
    assert result.shape == (50,)
    
    with pytest.raises(ValueError, match="Input expected to be a 1-d vector"):
        explainer.explain(X[:4], feature_idx=0)
        
    with pytest.raises(ValueError, match="Input expected to be a 1-d vector"):
        explainer.plot(X[:4], feature_idx=0)
            
def test_all_features(regressor):
    model, X, y = regressor

    explainer = CPExplainer(model, X)

    result = explainer.explain(
        X[0],
        feature_idx=None,
        n_grid=50
    )

    assert result.shape == (X.shape[1], 50)
    
def test_all_features_multiclass(multiclass_classifier):
    model, X, y = multiclass_classifier

    explainer = CPExplainer(model, X)

    result = explainer.explain(
        X[0],
        feature_idx=None,
        n_grid=50
    )

    assert result.shape == (X.shape[1], 50, len(np.unique(y))
    )