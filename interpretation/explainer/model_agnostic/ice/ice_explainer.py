from ..agnostic_explainer import AgnosticExplainer
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from ....utils.validate_input import validate_input_2d

class ICEExplainer(AgnosticExplainer):
    def __init__(self, input_model, input_data):
        super().__init__(input_model)
        self.data = input_data
    
    def explain(
        self, 
        X: npt.NDArray, 
        feature_idx: int = None, 
        n_grid: int = 50,
        centered: bool = True
    ) -> npt.NDArray:
        """Computes the ICE result for a single row

        Parameters
        ----------
        X : npt.NDArray
            Data point to explain.
        feature_idx : int
            Column index of the feature to explain.
        n_grid : int, optional
            Number of data points generated for the ICE explanation, by default 50.
        centered: bool, optional
            Centering so that all explanations start at the same point

        Returns
        -------
        npt.NDArray
            Model predictions for all ICE data points.

        Raises
        ------
        TypeError
            If X is not a numpy array.
        ValueError
            If X is not two-dimensional.
        """
                
        validate_input_2d(X)
        
        if feature_idx is not None:
            return self._compute_ice(X, feature_idx, n_grid, centered)
        else:
            ice_results = []
            for feature_idx in range(X.shape[-1]):
                result = self._compute_ice(X, feature_idx, n_grid, centered)
                ice_results.append(result)
            
            return np.array(ice_results)
                
            
    def plot(
        self, 
        X: npt.NDArray,
        feature_idx: int, 
        output_idx: int = 0, 
        n_grid: int = 50,
        ax: Axes = None,
        feature_name: str = None,
        centered: bool = True
    ) -> Axes:
        """Produce visual ICE plots for a given feature and output index

        Parameters
        ----------
        X : npt.NDArray
            All data points to explain.
        feature_idx : int
            Column index of the feature to explain.
        output_idx : int
            Since model may have multiple outputs (e.g. multi-class), an index of the output to explain can be specified.
        n_grid : int, optional
            Number of data points generated for the ICE explanation, by default 50.
        ax : Axes, optional
            Axes on which to draw the plot. If None, a new figure and axes are created., by default None
        feature_name : str, optional
            Name of the feature, used for labelling the x-axis, by default None
        centered: bool, optional
            Centering so that all explanations start at the same point

        Returns
        -------
        Axes
            The axes containing the ICE plot.

        Raises
        ------
        TypeError
            If X is not a numpy array.
        ValueError
            If X is not two-dimensional.
        """
        validate_input_2d(X)
        
        X_ice = self.explain(X, feature_idx, n_grid, centered)
        
        xj = self.data[:, feature_idx]
        min_xj, max_xj = xj.min(), xj.max()
        
        grid = np.linspace(min_xj, max_xj, n_grid, dtype=X.dtype)
        
        if ax is None:
            _, ax = plt.subplots()
        
        ax.plot(grid, X_ice[:, :, output_idx].T)
        ax.grid()
        ax.set_xlabel(feature_name)
        ax.set_ylabel("Prediction")
        ax.set_title("ICE Plot")
        
        return ax
        
    
    def _compute_ice(self, X, feature_idx, n_grid, centered):
        """Used for computing all model predictions for one varying feature for all data points"""
        xj = self.data[:, feature_idx]
        min_xj, max_xj = xj.min(), xj.max()
        
        grid = np.linspace(min_xj, max_xj, n_grid, dtype=X.dtype)
        
        n_samples = X.shape[0]
        X_ice = np.repeat(X, n_grid, axis=0)
        X_ice[:, feature_idx] = np.tile(grid, n_samples)
        
        pred = self.model(X_ice).reshape(n_samples, n_grid, -1)
        
        if pred.ndim == 1:
            pred = pred.reshape(-1, 1)
            
        if centered:
            pred = pred - pred[:, [0]]
            
        return pred