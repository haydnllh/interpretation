from ...explainer import Explainer
from ....models.wrapper.model_specific.linear_model import LinearModel
import numpy as np

class LinearExplainer(Explainer):
    def __init__(self, input_model):
        super().__init__(input_model)
        
        self.model = LinearModel(input_model)

    @property
    def coef(self):
        return self.model.coef
    
    @property
    def intercept(self):
        return self.model.intercept
    
    def SSE(self, X, y):
        pred = self.model(X)
        return np.sum(pred - y)
    
    def SST(self, y):
        mean = np.mean(y, axis=0)
        return np.sum(y - mean, axis=0)
    
    def R_squared(self, X, y, adjusted=True):
        n_samples, n_features = X.shape
        sse = self.SSE(X, y)
        sst = self.sst(y)
        r2 = 1 - sse / sst
        if adjusted:
            return 1 - (1 - r2) * ((n_samples - 1) / (n_samples - n_features - 1)) if adjusted else return r2
        else:
            return 1 - sse / sst