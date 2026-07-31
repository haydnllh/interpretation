from ..agnostic_explainer import AgnosticExplainer
import numpy as np
import numpy.typing as npt
from scipy.optimize import minimize
from scipy.stats import median_abs_deviation

class CounterfactualExplainer(AgnosticExplainer):
    def __init__(self, input_model, input_data, input_label):
        super().__init__(input_model)
        self.data = input_data
        self.mad = median_abs_deviation(input_data, axis=0)
        self.label = input_label
        
    def explain(
        self,
        X,
        loss_fn = None,
        lambda_initial=1e-2,
        lambda_max=1e4,
        lambda_multiplier=7,
        tol=None
    ) -> npt.NDArray:
        prediction_error = np.inf
        tol = 0.01 * np.std(self.label) if tol is None else tol
        lam = lambda_initial
        
        sample_idx = np.random.randint(0, len(self.data))
        cf = self.data[sample_idx].copy()
        y = self.label[sample_idx]
        
        while lam <= lambda_max and prediction_error > tol:
            cf = minimize(
                fun=self._loss if loss_fn is None else loss_fn,
                x0=cf,
                method="Nelder-Mead",
                args=(X, y, lam)
            ).x
            prediction_error = np.abs(self.model(cf.reshape(1, -1)) - y)
            lam *= lambda_multiplier
            
        return cf
        
        
    def _loss(self, cf, X, y, lam):
        prediction_error = (self.model(cf.reshape(1, -1)) - y) ** 2
        distance_error = np.sum(np.abs(X - cf) / self.mad)
        
        return float(lam * prediction_error + distance_error)
        