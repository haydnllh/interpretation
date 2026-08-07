import numpy as np
import numpy.typing as npt
from pymoo.core.problem import ElementwiseProblem

def wachter_loss(cf, X, y, lam, model, mad):
    """Loss function proposed by Wachter et al. (2018).
    L = λ(f̂(x') - y') ^ 2 + d(x, x'),
    where d(x, x') = L1 distance weighted by inverse MAD of each feature"""
    prediction_error = (model(cf.reshape(1, -1)) - y) ** 2
    distance_error = np.sum(np.abs(X - cf) / mad)
    
    return float(lam * prediction_error + distance_error)

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