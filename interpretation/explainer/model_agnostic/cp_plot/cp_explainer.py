from ..agnostic_explainer import AgnosticExplainer
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

class CPExplainer(AgnosticExplainer):
    def __init__(self, input_model, input_data):
        super().__init__(input_model)
        self.data = input_data
    
    def explain(
        self, 
        X: npt.NDArray, 
        feature_idx: int, 
        n_grid: int = 50
    ) -> npt.NDArray:
        """Computes the Ceteris Paribus (CP) result for a single row

        Parameters
        ----------
        X : npt.NDArray
            Data point to explain.
        feature_idx : int
            Column index of the feature to explain.
        n_grid : int, optional
            Number of data points generated for the CP explanation, by default 50.

        Returns
        -------
        npt.NDArray
            Model predictions for all CP data points.

        Raises
        ------
        TypeError
            If X is not a numpy array.
        ValueError
            If X is not a one-dimensional vector.
        """
                
        if not isinstance(X, np.ndarray):
            raise TypeError("Input must be an instance of np.ndarray")
        
        if X.ndim > 1 and all(s == 1 for s in X.shape[:-1]):
            raise ValueError("Input expected to be a 1-d vector")
        X = X.reshape(-1)
        
        if feature_idx is not None:
            return self._compute_cp(X, feature_idx, n_grid)
        else:
            cp_results = []
            for feature_idx in range(X.shape[-1]):
                result = self._compute_cp(X, feature_idx, n_grid)
                cp_results.append(result)
            
            return np.array(cp_results)
                
            
    def plot(
        self, 
        X: npt.NDArray,
        feature_idx: int, 
        output_idx: int = 0, 
        n_grid: int = 50,
        ax: Axes = None,
        feature_name: str = None
    ) -> Axes:
        """Produce visual CP plots for a given feature and output index

        Parameters
        ----------
         X : npt.NDArray
            Data point to explain.
        feature_idx : int
            Column index of the feature to explain.
        output_idx : int
            Since model may have multiple outputs (e.g. multi-class), an index of the output to explain can be specified.
        n_grid : int, optional
            Number of data points generated for the CP explanation, by default 50.
        ax : Axes, optional
            Axes on which to draw the plot. If None, a new figure and axes are created., by default None
        feature_name : str, optional
            Name of the feature, used for labelling the x-axis, by default None

        Returns
        -------
        Axes
            The axes containing the Ceteris Paribus plot.

        Raises
        ------
        TypeError
            If X is not a numpy array.
        ValueError
            If X is not a one-dimensional vector.
        """
        if not isinstance(X, np.ndarray):
            raise TypeError("Input must be an instance of np.ndarray")
        
        if X.ndim > 1 and all(s == 1 for s in X.shape[:-1]):
            raise ValueError("Input expected to be a 1-d vector")
        
        X_cp = self.explain(X, feature_idx, n_grid)
        
        xj = self.data[:, feature_idx]
        min_xj, max_xj = xj.min(), xj.max()
        
        grid = np.linspace(min_xj, max_xj, n_grid, dtype=X.dtype)
        
        y = self.model(np.expand_dims(X, 0))
        
        if ax is None:
            _, ax = plt.subplots()
        
        ax.plot(grid, X_cp[:, output_idx], zorder=1)
        ax.scatter(X[feature_idx], y, c="red", zorder=10)
        ax.grid()
        ax.set_xlabel(feature_name)
        ax.set_ylabel("Prediction")
        ax.set_title("Ceteris Paribus Plot")
        
        return ax
        
    
    def _compute_cp(self, X, feature_idx, n_grid):
        """Used for computing all model predictions for one varying feature"""
        xj = self.data[:, feature_idx]
        min_xj, max_xj = xj.min(), xj.max()
        
        grid = np.linspace(min_xj, max_xj, n_grid, dtype=X.dtype)
        X_cp = np.vstack((X, ) * n_grid, dtype=X.dtype)
        X_cp[:, feature_idx] = grid
        
        pred = self.model(X_cp)
        return pred