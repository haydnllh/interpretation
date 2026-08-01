from ..specific_explainer import SpecificExplainer
from ....models.wrapper.model_specific.linear_model import LinearModel
import numpy as np
import numpy.typing as npt
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
from ....utils.validate_input import validate_input_label, validate_input_2d

class LinearExplainer(SpecificExplainer):
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
        return np.sum((pred - y) ** 2, axis=0)
    
    def SST(self, y):
        mean = np.mean(y, axis=0)
        return np.sum((y - mean) ** 2, axis=0)
    
    def R_squared(
        self, 
        X: npt.NDArray, 
        y: npt.NDArray, 
        adjusted: bool=True
    ):
        """Computes R-squared metric

        Parameters
        ----------
        X : npt.NDArray
            Input feature matrix of shape (n_samples, n_features).
        y : npt.NDArray
            Target values of shape (n_samples,) or (n_samples, n_output).
        adjusted : bool, optional
            Whether to compute the adjusted R-squared, by default True.

        Returns
        -------
        float or npt.NDArray
            The R-squared value(s). Returns a float for single-output models
            or an array of shape (n_outputs,) for multi-output models.
        """
        
        validate_input_label(X, y)
        
        n_samples, n_features = X.shape
        sse = self.SSE(X, y)
        sst = self.SST(y)
        r2 = 1 - sse / sst
        if adjusted:
            return 1 - (1 - r2) * ((n_samples - 1) / (n_samples - n_features - 1)) if adjusted else r2
        else:
            return r2
        
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
        
        n_samples, n_features = X.shape
        pred = self.model(X)
        
        X_design = np.column_stack([np.ones(n_samples), X])
        sigma_squared = np.sum((pred - y) ** 2, axis=0) / (n_samples - n_features - 1)
        XTX_inv = np.linalg.inv(X_design.T @ X_design)[None, :, :]
        
        if np.ndim(sigma_squared) == 0:
            covariance = sigma_squared * XTX_inv
        else:
            covariance = sigma_squared[:, None, None] * XTX_inv
        
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
    
    def weight_plot(
        self,
        X: npt.NDArray,
        y: npt.NDArray,
        ax: Axes,
        output_idx: int = 0,
        feature_names: list[str] = None
    ):
        """Produce the weight plot for one output dimension, showing the 95% confidence intervals on model coefficients.

        Parameters
        ----------
        X : npt.NDArray
            Input feature matrix of shape (n_samples, n_features).
        y : npt.NDArray
            Target values of shape (n_samples,) or (n_samples, n_output).
        ax : Axes
            Axes on which to draw the plot. If None, a new figure and axes are created., by default None.
        output_idx : int, optional
            An output index can be specified if model has multiple outputs, by default 0
        feature_names : list[str], optional
            Name of the features, used for labelling the y-axis, by default None

        Returns
        -------
        Axes
            The axes containing the weight plot.
        """
        
        validate_input_label(X, y)
        
        if ax is None:
            _, ax = plt.subplots()
            
        t_value = 1.96
        margin = t_value * self.standard_error(X, y)
        margin = margin[..., 1:]
        
        coef = self.coef
        if coef.ndim == 2:
            coef = coef[output_idx]
            margin = margin[output_idx]
        feature_pos = np.arange(len(coef))
        
        ax.errorbar(
            coef,
            feature_pos,
            xerr=margin,
            fmt='o',
            linestyle='none'
        )
        ax.grid()
        ax.set_xlabel("Weight estimate")
        ax.set_yticks(feature_pos, feature_names)
        ax.set_title("Weight plot")
        ax.axvline(0, linestyle="--")
        
        return ax
    
    def effect_plot(
        self,
        X: npt.NDArray,
        ax: Axes,
        output_idx: int = 0,
        feature_names: list[str] = None
    ):
        """Produce the effect plot for one output dimension, showing the box plot of the weights' contributions.
        
        Parameters
        ----------
        X : npt.NDArray
            Input feature matrix of shape (n_samples, n_features).
        ax : Axes
            Axes on which to draw the plot. If None, a new figure and axes are created., by default None.
        output_idx : int, optional
            An output index can be specified if model has multiple outputs, by default 0
        feature_names : list[str], optional
            Name of the features, used for labelling the y-axis, by default None

        Returns
        -------
        Axes
            The axes containing the effect plot.
        """
        
        validate_input_2d(X)
        
        coef = self.coef
        if coef.ndim == 2:
            coef = coef[output_idx]
        feature_pos = np.arange(1, len(coef) + 1)
        
        effects = X * coef
        
        ax.boxplot(
            effects,
            vert=False,
            positions=feature_pos
        )
        ax.set_xlabel("Feature effect")
        ax.set_yticks(feature_pos, feature_names)
        ax.set_title("Effect plot")
        ax.grid()
        ax.axvline(0, linestyle="--")
        
        return ax
        