from ..specific_explainer import SpecificExplainer
from ....models.wrapper.model_specific.logistic_model import LogisticModel
import numpy as np
import numpy.typing as npt
from matplotlib.axes import Axes
from sklearn.decomposition import PCA
from ....utils.validate_input import validate_input_label

class LogisticExplainer(SpecificExplainer):
    def __init__(self, input_model):
        super().__init__(input_model)
        
        self.model = LogisticModel(input_model)

    @property
    def coef(self):
        return self.model.coef
    
    @property
    def intercept(self):
        return self.model.intercept
    
    def odds(self):
        return np.exp(self.coef)
    
    def standard_error(
        self,
        X: npt.NDArray,
        y: npt.NDArray
    ) -> npt.NDArray:
        """Computes the standard error of the coefficients

        Parameters
        ----------
        X : npt.NDArray
            Input feature matrix of shape (n_samples, n_features).
        y : npt.NDArray
            Target values of shape (n_samples,) or (n_samples, n_output).

        Returns
        -------
        npt.NDArray
            Standard error of coefficients (n_features) or (n_output, n_features).
        """
        
        validate_input_label(X, y)
        
        n_samples, _ = X.shape
        pred = self.model(X)
        
        X_design = np.column_stack([np.ones(n_samples), X])
        W = pred * (1 - pred)
        
        if W.ndim == 1:
            W = W[None, :]
        
        fisher_information = np.einsum(
            "ni,no,nj->oij",
            X_design,
            W,
            X_design
        ) # computes the fisher information matrix X^T @ diag(W) @ X
        covariance = np.linalg.pinv(fisher_information)
        
        se = np.sqrt(np.diagonal(covariance, axis1=-2, axis2=-1))
        return se.squeeze()
    
    def t_statistic(
        self,
        X: npt.NDArray,
        y: npt.NDArray
    ) -> npt.NDArray:
        """Computes the t-statistic feature importance score for each coefficient

        Parameters
        ----------
        X : npt.NDArray
            Input feature matrix of shape (n_samples, n_features).
        y : npt.NDArray
            Target values of shape (n_samples,) or (n_samples, n_output).
            
        Returns
        -------
        npt.NDArray
            The t-statistic feature importance of shape (n_features) or (n_output, n_features)
        """
        
        validate_input_label(X, y)
        
        se = self.standard_error(X, y)[..., 1:]
        
        beta = self.coef
        t = beta[None, :] / se
        return t.squeeze()
    
    def decision_boundary(
        self,
        X: npt.NDArray,
        y: npt.NDArray,
        n_grid: int = 50,
        ax: Axes = None
    ) -> Axes:
        """Plot the coloured decision boundary of the model in addition to scatter plot of the input X and y. 

        Parameters
        ----------
        X : npt.NDArray
            Input feature matrix of shape (n_samples, n_features).
        y : npt.NDArray
            Target values of shape (n_samples,) or (n_samples, n_output).
        n_grid : int, optional
            Number of grid points generated along each plot axis., by default 50
        ax : Axes, optional
            Axes to draw the decision boundary, by default None

        Returns
        -------
        Axes
            The plot of decision boundary.
        """
        
        validate_input_label(X, y)
        
        _, n_features = X.shape
        pca = PCA(n_components=2)
        
        if n_features != 2:
            X_data = pca.fit_transform(X)
        else:
            X_data = X
        
        x_min, x_max = X_data[:, 0].min() - 1, X_data[:, 0].max() + 1
        y_min, y_max = X_data[:, 1].min() - 1, X_data[:, 1].max() + 1
        
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, n_grid), np.linspace(y_min, y_max, n_grid))
        X_grid = np.column_stack([xx.ravel(), yy.ravel()])
        if n_features != 2:
            pred = self.model(pca.inverse_transform(X_grid))
        else: 
            pred = self.model(X_grid)
        pred = pred.reshape(xx.shape)
        
        ax.scatter(X_data[:, 0], X_data[:, 1], c=y, cmap="Paired")
        ax.contourf(xx, yy, pred, alpha=0.3, cmap="Paired")
        
        if n_features != 2:
            print(n_features, "hi")
            ax.set_xlabel("PCA axis 1")
            ax.set_ylabel("PCA axis 2")
        else:
            ax.set_xlabel(r"$X_1$")
            ax.set_ylabel(r"$X_2$")
        return ax