from ..agnostic_explainer import AgnosticExplainer
import numpy as np
import numpy.typing as npt
from scipy.optimize import minimize
from scipy.stats import median_abs_deviation

class CounterfactualExplainer(AgnosticExplainer):
    """Counterfactual explainer using methods proposed by Wachter et al. (2018)."""
    def __init__(self, input_model, input_data):
        """Initialise the explainer by taking in the model to explain
        and the observed data to be used for to estimate the MAD and sampling."""
        super().__init__(input_model)
        self.data = input_data
        self.mad = median_abs_deviation(input_data, axis=0)
        
    def explain(
        self,
        X: npt.NDArray,
        desired_y: npt.NDArray,
        lambda_initial: float = 1e-2,
        lambda_max: float = 1e4,
        lambda_multiplier: float = 10.0,
        tol: float = 1e-3
    ) -> npt.NDArray:
        """Computes a counterfactual that produces a model output specified by desired_y.

        Parameters
        ----------
        X : npt.NDArray
            The instance to be explained by the counterfactual.
        desired_y : npt.NDArray
            The predefined prediction for the counterfactual explanation to produce.
        lambda_initial : float, optional
            Initial lambda, by default 1e-2
        lambda_max : float, optional
            Maximum lambda, by default 1e4
        lambda_multiplier : float, optional
            Multiplier of lambda after each iteration, by default 10
        tol : float, optional
            Tolerance for prediction error, the function increases lambda exponentially if prediction error is higher than tol, by default 1e-3.

        Returns
        -------
        npt.NDArray
            Counterfactual instance.
        """
        prediction_error = np.inf
        lam = lambda_initial
        
        sample_idx = np.random.randint(0, len(self.data))
        cf = self.data[sample_idx].copy()
        
        while lam <= lambda_max and prediction_error > tol:
            cf = minimize(
                fun=self._loss,
                x0=cf,
                method="Nelder-Mead",
                args=(X, desired_y, lam)
            ).x
            prediction_error = np.abs(float(self.model(cf.reshape(1, -1))) - desired_y)
            lam *= lambda_multiplier
            
        return cf
        
        
    def _loss(self, cf, X, y, lam):
        """Loss function proposed by Wachter et al. (2018).
        L = λ(f̂(x') - y') ^ 2 + d(x, x'),
        where d(x, x') = L1 distance weighted by inverse MAD of each feature"""
        prediction_error = (self.model(cf.reshape(1, -1)) - y) ** 2
        distance_error = np.sum(np.abs(X - cf) / self.mad)
        
        return float(lam * prediction_error + distance_error)
        