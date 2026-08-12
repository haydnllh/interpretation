from ..agnostic_explainer import AgnosticExplainer
from .permutation import generate_permutations
import numpy as np
import numpy.typing as npt 
import math

class SHAPExplainer(AgnosticExplainer):
    def __init__(self, input_model, input_data):
        super().__init__(input_model)
        self.data = input_data
    
    def explain(
        self,
        X:npt.NDArray,
        n_permutation,
        n_sample
    ):
        n_features = len(X)
        n_rows = self.data.shape[0]
        n_sample = min(n_rows, n_sample)
        if n_features < 21:
            n_permutation = min(n_permutation, math.factorial(n_features))

        sample_idx = np.random.choice(self.data.shape[0], size=n_sample, replace=False)
        samples = self.data[sample_idx]  
        
        perm_sample, perm_idx = generate_permutations(
            X,
            samples,
            n_permutation,
        )
        
        pred_perm = self.model(perm_sample.reshape(-1, n_features))
        pred_perm = pred_perm.reshape(n_permutation, n_features, n_sample).mean(axis=2)

        pred_mean = self.model(samples).mean()
        pred_perm = np.insert(pred_perm, 0, pred_mean, axis=1)

        phis = np.zeros(n_features)

        for p_perm, p_idx in zip(pred_perm, perm_idx):
            contributions = np.diff(p_perm)
            phis[p_idx] += contributions
            
        phis /= n_permutation
        return phis