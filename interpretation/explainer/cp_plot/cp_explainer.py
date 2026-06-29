from ..explainer import Explainer
import numpy as np
import numpy.typing as npt

class CPExplainer(Explainer):
    def __init__(self, input_model, input_data):
        super().__init__(input_model)
        self.data = input_data
    
    def explain(
        self, 
        X: npt.NDArray, 
        feature_idx: int = None, 
        n_grid: int = 50
    ) -> npt.NDArray:
        """Computes the Ceteris Paribus (CP) result for a single row."""
        
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
                
            
    def plot(self):
        pass
    
    def _compute_cp(self, X, feature_idx, n_grid):
        xj = self.data[:, feature_idx]
        min_xj, max_xj = xj.min(), xj.max()
        
        grid = np.linspace(min_xj, max_xj, n_grid)
        X_explain = np.vstack((X, ) * n_grid)
        X_explain[:, feature_idx] = grid
        
        pred = self.model(X_explain)
        return pred