from ..agnostic_explainer import AgnosticExplainer
from ....utils.validate_input import validate_input_1d
import numpy as np
import numpy.typing as npt
from scipy.optimize import minimize as spminimize
from scipy.stats import median_abs_deviation
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.termination import get_termination

class CounterfactualExplainer(AgnosticExplainer):
    """Counterfactual explainer using methods proposed by Wachter et al. (2018)."""
    def __init__(self, input_model, input_data):
        """Initialise the explainer by taking in the model to explain
        and the observed data to be used for to estimate the MAD and sampling."""
        super().__init__(input_model)
        self.data = input_data
        self.mad = median_abs_deviation(input_data, axis=0)
        self.safe_mad = np.where(self.mad == 0, 1.0, self.mad)
        self.xl = np.min(input_data, axis=0)
        self.xu = np.max(input_data, axis=0)
        self.r = self.xu - self.xl
        self.safe_r = np.where(self.r == 0, 1.0, self.r)
        
        
    def explain(
        self,
        X: npt.NDArray,
        desired_y: float,
        method: str = "wachter",
        **method_kwargs
    ) -> npt.NDArray:
        """Computes a counterfactual that produces a model output specified by desired_y.
        Only supports scalar labels.

        Parameters
        ----------
        X : npt.NDArray
            The instance to be explained by the counterfactual, (n_features).
        desired_y : float
            The predefined prediction for the counterfactual explanation to produce.
        method: str
            The method used to compute the counterfactual, either 'wachter' or 'dandl', by default 'wachter'.
            
            - 'wachter': uses the optimisation method proposed by Wachter et al. (2018).
            - 'dandl': uses the multi-objective + NSGA optimisation method proposed by Dandl et al. (2020).
        **method_kwargs
            Additional keyword arguments passed directly to the selected
            counterfactual generation method. See the documentation of
            :meth:`wachter` and :meth:`dandl` for the available options.

        Returns
        -------
        npt.NDArray
            One Counterfactual instance for wachter and multiple instances for dandl.
            
        Raises
        ------
        ValueError
            If ``method`` is not one of the supported methods.
        """
        
        validate_input_1d(X)
        X = X.reshape(-1)
        
        if method == "wachter":
            cf = self.wachter(X, desired_y, **method_kwargs)
        elif method == "dandl":
            cf = self.dandl(X, desired_y, **method_kwargs)
        else:
            raise ValueError(f"Unknown method {method}. Expected to be: 'wachter' or 'dandl'")
            
        return cf
        
        
    def _wachter_loss(self, cf, X, y, lam):
        """Loss function proposed by Wachter et al. (2018).
        L = λ(f̂(x') - y') ^ 2 + d(x, x'),
        where d(x, x') = L1 distance weighted by inverse MAD of each feature"""
        prediction_error = (self.model(cf.reshape(1, -1)) - y) ** 2
        distance_error = np.sum(np.abs(X - cf) / self.safe_mad)
        
        return float(lam * prediction_error + distance_error)
    
    def wachter(
        self,
        X: npt.NDArray,
        desired_y: float,
        lambda_initial: float = 1e-2,
        lambda_max: float = 1e4,
        lambda_multiplier: float = 10.0,
        tol: float = 1e-3
    ):
        """Computes a counterfactual using method proposed by Wachter et al.

        Parameters
        ----------
        X : npt.NDArray
            The instance to be explained by the counterfactual, (n_features).
        desired_y : float
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
        bounds = list(zip(
            self.xl,
            self.xu
        ))
        
        sample_idx = np.random.randint(0, len(self.data))
        cf = self.data[sample_idx].copy()
        
        while lam <= lambda_max and prediction_error > tol:
            cf = spminimize(
                fun=self._wachter_loss,
                x0=cf,
                method="Nelder-Mead",
                args=(X, desired_y, lam),
                bounds=bounds
            ).x
            prediction_error = np.abs(float(self.model(cf.reshape(1, -1))) - desired_y)
            lam *= lambda_multiplier
            
        return cf
    
    def dandl(
        self, 
        X: npt.NDArray, 
        desired_y: float, 
        k: int = 5,
        topn: int = 10
    ):
        """Computes a counterfactual via nondominated sorting genetic algorithm.

        Parameters
        ----------
        X : npt.NDArray
            The instance to be explained by the counterfactual, (n_features).
        desired_y : float
            The predefined prediction for the counterfactual explanation to produce.
        k : int, optional
            Number of samples taken to take average in fourth objective, k > 0, by default 5.
        topn: int, optional
            Top n counterfactual to return, by default 10.

        Returns
        -------
        npt.NDArray
            Counterfactual instances with shape (n_candidates, n_features).
        """
        problem = self.FourCriteriaLoss(
            X, 
            desired_y, 
            self.data, 
            self.model, 
            self.safe_r, 
            self.xl,
            self.xu,
            k
        )
        
        algorithm = NSGA2(
            pop_size=100, 
            eliminate_duplicates=True
        )
        
        result = minimize(
            problem,
            algorithm,
            termination=get_termination("n_gen", 200),
        )
        
        prediction_errors = np.abs(self.model(result.X) - desired_y)
        top_idx = np.argsort(prediction_errors)
        cfs = result.X[top_idx[:topn]]
        
        return cfs
        
    
    class FourCriteriaLoss(ElementwiseProblem):
        """
        The 4 objectives proposed by Dandl et al. (2020).
        o1 = |f̂(x') - y'|
        o2 = (1 / n_features) * sum of all feature Gower distance between x' and x
        o3 = ||x - x'||_0, L0 norm
        o4 = (1 / n_features) * average of the sum of all feature Gower distance between x' and k nearest points
        """
        def __init__(self, X, desired_y, data, model, r, xl, xu, k=5):
            super().__init__(
                n_var=data.shape[-1], 
                n_obj=4,
                xl=xl,
                xu=xu
            )
            
            if not isinstance(k, int) or k < 1:
                raise ValueError("k must be a positive non-zero integer")
            
            self.X = X
            self.desired_y = desired_y
            self.data = data
            self.model = model
            self.r = r
            self.k = k
            
        def _evaluate(self, cf, out):
            pred = self.model(cf.reshape(1, -1))
            gower_o4 = np.sum(np.abs(self.data - cf[None, :]) / self.r, axis=1)
            k = min(self.k, len(gower_o4))
            sparse_tol = 1e-8
            
            o1 = float(np.abs((pred - self.desired_y))) # L1 norm
            o2 = float(np.sum(np.abs(self.X - cf) / self.r) / cf.shape[-1]) # Gower distance
            o3 = float(np.sum(np.abs(self.X - cf) > sparse_tol)) # L0 norm
            o4 = float(np.mean(np.partition(gower_o4, k - 1)[:k]) / cf.shape[-1]) # Average Gower distance
            
            out["F"] = np.array([o1, o2, o3, o4])