from ..agnostic_explainer import AgnosticExplainer
from .sampling import gaussian_samples
from .kernel import gaussian_RBF
import numpy as np
import numpy.typing as npt
from sklearn.linear_model import Ridge
from ....utils.validate_input import validate_input_1d

class LIMEExplainer(AgnosticExplainer):
    """Only works for continuous tabular data"""
    def __init__(self, input_model, input_data):
        """Initialise the explainer by taking in the model to explain and the observed data to be used for sampling."""
        super().__init__(input_model)
        
        self.sigma = np.std(input_data, axis=0)
    
    def explain(
        self, 
        X,
        n_samples=100,
    ) -> npt.NDArray:
        """Takes in the data point to explain and returns the weights and bias of the local Ridge model.

        Parameters
        ----------
        X : npt.NDArray
            Data point to explain, (n_features).
        n_samples : int, optional
            Number of samples to generate for the local model, by default 100.

        Returns
        -------
        npt.NDArray
            An array containing the weights of the local Ridge model with the last element as the bias, (n_features + 1).
        """
        
        validate_input_1d(X)
        X = X.reshape(-1)
        
        n_features = X.shape
        
        X_samples = gaussian_samples(X, n_samples, self.sigma)
        y_samples = self.model(X_samples)
        weights = gaussian_RBF(X, X_samples, 0.75 * np.sqrt(n_features))
        
        simple_model = Ridge(alpha=1.0)
        simple_model.fit(X_samples, y_samples, sample_weight=weights)
        explanations = np.append(simple_model.coef_, simple_model.intercept_)
        
        return explanations