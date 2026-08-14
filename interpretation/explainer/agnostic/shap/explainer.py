from ..agnostic_explainer import AgnosticExplainer
from ....utils.validate_input import validate_input_2d
from .permutation import generate_permutations
import numpy as np
import numpy.typing as npt 
import math
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

class SHAPExplainer(AgnosticExplainer):
    """
    SHAP explainer estimates the Shapley values of a given instance.
    Only support tabular data and model output with dimension 1 at the moment.
    """
    def __init__(self, input_model, input_data):
        super().__init__(input_model)
        self.data = input_data
    
    def explain(
        self,
        X:npt.NDArray,
        n_permutations:int = 10,
        n_samples:int = 100
    ) -> tuple[npt.NDArray, npt.NDArray]:
        """Estimate Shapley values for input instances using permutation sampling.

        Parameters
        ----------
        X : npt.NDArray
            The instances to be explained by the Shapley values, (n_instances, n_features).
        n_permutations : int, optional
            The number of permutations to sample for estimating Shapley values, by default 10.
        n_samples : int, optional
            The number of background reference samples to draw from `input_data` to marginalise out missing features, by default 100.

        Returns
        -------
        phis_matrix : npt.NDArray
            A 2D array of estimated Shapley values ($\phi$) representing the contribution of each feature in `X` to the model's prediction, (n_instances, n_features).
        pred_mean : npt.NDArray
            The baseline model prediction used to compute the Shapley values.
        """
        
        X = validate_input_2d(X)
        
        n_instances, n_features = X.shape
        n_rows = self.data.shape[0]
        n_samples = min(n_rows, n_samples)
        if n_features < 21:
            n_permutations = min(n_permutations, math.factorial(n_features))

        sample_idx = np.random.choice(self.data.shape[0], size=n_samples, replace=False)
        samples = self.data[sample_idx]  
        pred_mean = self.model(samples).mean()
        
        phis_matrix = np.zeros_like(X, dtype=float)
        
        for i in range(n_instances):
            X_inst = X[i]
            perm_sample, perm_idx = generate_permutations(
                X_inst,
                samples,
                n_permutations,
            )
            
            pred_perm = self.model(perm_sample.reshape(-1, n_features))
            pred_perm = pred_perm.reshape(n_permutations, n_features, n_samples).mean(axis=2)

            pred_perm = np.insert(pred_perm, 0, pred_mean, axis=1)

            for p_perm, p_idx in zip(pred_perm, perm_idx):
                contributions = np.diff(p_perm)
                phis_matrix[i, p_idx] += contributions
            
        phis_matrix /= n_permutations
        return phis_matrix, pred_mean

    def beeswarm(
        self,
        X: npt.NDArray,
        n_permutations: int = 10,
        n_samples: int = 100,
        feature_names: list[str] | None = None,
        max_n_features: int = 10,
        ax: plt.Axes | None = None
    ) -> tuple[plt.Figure, plt.Axes]:
        n_instances, n_features = X.shape
        n_display_features = min(max_n_features, n_features)
        
        phis_matrix, _ = self.explain(X, n_permutations, n_samples)
        mean_abs_phi = np.mean(np.abs(phis_matrix), axis=0)
        ranked_idx = np.argsort(mean_abs_phi)[::-1][:max_n_features]
        ranked_idx = ranked_idx[::-1]
        phis_matrix_ranked = phis_matrix[:, ranked_idx]
        X_ranked = X[:, ranked_idx]
        
        if feature_names is None:
            feature_names = [f"Feature {i}" for i in range(n_features)]
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, max(4, len(ranked_idx) * 0.5)))
        else:
            fig = ax.figure
            
        shap_cmap = LinearSegmentedColormap.from_list("shap_cmap", ["#008bfb", "#ff0051"])
        
        y = np.tile(np.arange(n_display_features), (n_instances, 1))
        jitter = np.random.uniform(-0.15, 0.15, size=(n_instances, n_display_features))
        y = y + jitter
        
        fmax, fmin = np.max(X_ranked, axis=0)[None, :], np.min(X_ranked, axis=0)[None, :]
        diff = fmax - fmin
        diff[diff == 0] = 1.0
        f_norm = (X_ranked - fmin) / diff
        
        scatter = ax.scatter(
            phis_matrix_ranked.ravel(),
            y.ravel(),
            c=f_norm.ravel(),
            cmap=shap_cmap,
            s=10,
            linewidths=0,
            zorder=10
        )
        ax.set_yticks(range(len(ranked_idx)))
        ax.set_yticklabels([feature_names[i] for i in ranked_idx])
        ax.set_xlabel("SHAP Value (feature contribution)")
        ax.set_title("SHAP Beeswarm Plot",fontweight="bold", pad=12)
        ax.axvline(0, color="lightgrey", linewidth=1, alpha=0.5, zorder=1)
        
        cbar = fig.colorbar(scatter, ax=ax, aspect=25, pad=0.04)
        cbar.set_label("Feature Value", rotation=270, labelpad=15, fontsize=10)
        cbar.set_ticks([0, 1])
        cbar.set_ticklabels(["Low", "High"])
        
        return fig, ax