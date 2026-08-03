from ..agnostic_explainer import AgnosticExplainer
from utils.validate_input import validate_input_1d
import numpy as np
import numpy.typing as npt
from typing import Sequence

class AnchorExplainer(AgnosticExplainer):
    def __init__(self, input_model, input_data):
        super().__init__(input_model)
        self.data = input_data
        self.lq = np.quantile(input_data, 0.25, axis=0)
        self.median = np.median(input_data, axis=0)
        self.uq = np.quantile(input_data, 0.75, axis=0)
        
    def explain(
        self,
        X: npt.NDArray,
        threshold: float,
        categorical_features: Sequence[int] = [],
        n_samples: int = 1000
    ):
        validate_input_1d(X)
        X = X.reshape(-1)
        
        n_samples = min(n_samples, len(self.data))
        samples_idx = np.random.sample(np.arange(0, len(self.data)), size=(n_samples), replace=False)
        X_samples = self.data[samples_idx]
        precisions = []
        
        for i in range(len(X)):
            precisions.append(self._precision(X_samples))
    
    def _precision(
        self,
        X_samples,
        predicate
    ):
        return predicate(X_samples) / len(X_samples)
    
    def _generate_predicates(
        self,
        X
    ):
        pass