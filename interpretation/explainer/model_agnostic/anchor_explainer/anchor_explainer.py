from ..agnostic_explainer import AgnosticExplainer
from ....utils.validate_input import validate_input_1d
import numpy as np
import numpy.typing as npt
from typing import Sequence

class AnchorExplainer(AgnosticExplainer):
    """Produces the anchors of a classifier given a sample using a greedy algorithm."""
    def __init__(self, input_model, input_data, is_prob):
        """
        Parameters
        ----------
        input_model
            Model to be explained.
        input_data
            Reference data for sampling and predicate generation, (n_samples, n_features).
        is_prob : bool
            Whether the model outputs class probabilities. If True, the predicted class is
            obtained is the class with the highest probability.
            If False, the model outputs are treated as the predicted class directly.
        """
        super().__init__(input_model)
        self.data = input_data
        self.lq = np.quantile(input_data, 0.25, axis=0)
        self.median = np.median(input_data, axis=0)
        self.uq = np.quantile(input_data, 0.75, axis=0)
        self.is_prob = is_prob
        
    def explain(
        self,
        X: npt.NDArray,
        threshold: float,
        categorical_features: Sequence[int] = None,
        n_samples: int = 1000,
        beam_width: int = 3
    ) -> list[dict]:
        """Computes the anchor for X using greedy algorithm.

        Parameters
        ----------
        X : npt.NDArray
           The instance to be explained by the anchors, (n_features).
        threshold : float
            The precision threshold of the anchor.
        categorical_features : Sequence[int], optional
            A list of categorical features indices, by default None
        n_samples : int, optional
            Number of samples to be drawn from `input_data` to calculate precision and coverage, by default 1000
        beam_width : int, optional
            Beam width for beam search of optimal anchor, by default 3

        Returns
        -------
        list[dict]
            Returns the anchor containing a list of predicates represented as dictionaries.
            
            For non-categorical features, the dictionary has the following example's format:
            {
                'idx': 0, # index of the feature
                'categorical': False, # not categorical
                'lower': 1.0, # lower bound of the predicate
                'upper': 2.0 # upper bound of the predicate
            }
            This predicate represents '1.0 < x_0 <= 2.0'
            
            For categorical features, the dictionary has the following example's format:
            {
                'idx': 0, # index of the feature
                'categorical': True, # is categorical
                'value': 2 # the category of the predicate
            }
            This predicate represents 'x_0 == 2'.
        
        Raises
        ------
        ValueError
            ``threshold`` must be a value between 0 and 1.
        ValueError
            ``n_samples`` must be a positive non-zero integer.
        ValueError
            ``beam_width`` must be a positive non-zero integer.
        """
        validate_input_1d(X)
        X = X.reshape(-1)
        
        if not 0 <= threshold <= 1:
            raise ValueError("threshold must be between 0 and 1.")
        if n_samples <= 0 or not isinstance(n_samples, int):
            raise ValueError("n_samples must be a positive non-zero integer.")
        if not isinstance(beam_width, int) or beam_width <= 0:
            raise ValueError("beam_width must be a positive non-zero integer.")
        
        if categorical_features is None:
            categorical_features = []
        
        n_samples = min(n_samples, len(self.data))
        samples_idx = np.random.choice(np.arange(0, len(self.data)), size=(n_samples), replace=False)
        X_samples = self.data[samples_idx]
        predicates = self._generate_predicates(X, categorical_features)
        result_predicates = []
        beam = [()]
        
        for _ in range(len(X)):
            candidates = []
            for anchor_indices in beam:
                start = anchor_indices[-1] + 1 if anchor_indices else 0
                for predicate_idx in range(start, len(predicates)):
                    new_indices = anchor_indices + (predicate_idx,)
                    new_anchor = [predicates[idx] for idx in new_indices]
                    
                    precision, coverage = self._precision_coverage(
                        X,
                        X_samples,
                        new_anchor
                    )
                    candidates.append((new_indices, new_anchor, precision, coverage))
            
            valid_candidates = [
                candidate
                for candidate in candidates
                if candidate[2] >= threshold
            ]
            
            if valid_candidates:
                _, optimal_anchor, _, _ = max(
                    valid_candidates,
                    key = lambda candidate: (
                        candidate[3],
                        candidate[2]
                    )
                )
                return optimal_anchor
            else:
                top_candidates = sorted(
                    candidates,
                    key = lambda candidate: (
                        candidate[2],
                        candidate[3]
                    )
                )[-beam_width:]
                beam = [c[0] for c in top_candidates]
                    
        optimal_anchor = [predicates[idx] for idx in beam[-1]]
        return optimal_anchor
    
    def _precision_coverage(
        self,
        X,
        X_samples,
        anchor,
    ):
        """Computes the precision and coverage for an anchor"""
        pred_X = self.model(X.reshape(1, -1))
        pred_samples = self.model(X_samples)
        mask = np.ones(X_samples.shape[0], dtype=bool)
        
        binary = (pred_samples.ndim == 2 and pred_samples.shape[-1] == 1) \
            or (pred_samples.ndim == 1)
        
        if self.is_prob and binary:
            label_X = pred_X > 0.5
            label_samples = (pred_samples > 0.5).astype(int)
        elif self.is_prob and not binary:
            label_X = np.argmax(pred_X, axis=1)
            label_samples = np.argmax(pred_samples, axis=1)
        else:
            label_X, label_samples = pred_X, pred_samples
        label_X = int(label_X)
        label_samples = label_samples.astype(int).reshape(-1)
        
        for p in anchor:
            feature = X_samples[:, p["idx"]]
            if p["categorical"]:
                predicate_mask = feature == p["value"]
            else:
                predicate_mask = (feature > p["lower"]) & (feature <= p["upper"])
            mask &= predicate_mask
            
        if not np.any(mask):
            return 0.0, 0.0
        
        precision = np.mean(label_samples[mask] == label_X)
        coverage = np.mean(mask)
        
        return precision, coverage
        
    
    def _generate_predicates(
        self,
        X,
        categorical_features,
    ):
        """Generate all predicates for a given instance, categorical or non-categorical."""
        ps = []
        
        for i in range (len(X)):
            if i in categorical_features:
                ps.append({
                    "idx": i,
                    "categorical": True,
                    "value": X[i]
                })
            else:
                lq, m, uq = self.lq[i], self.median[i], self.uq[i]
                quantiles = np.array([lq, m ,uq])
                idx_quantile = np.searchsorted(quantiles, X[i], side="left")
                
                match idx_quantile:
                    case 0:
                        lower, upper = -np.inf, lq
                    case 1:
                        lower, upper = lq, m
                    case 2:
                        lower, upper = m, uq
                    case 3:
                        lower, upper = uq, np.inf
                        
                ps.append({
                    "idx": i,
                    "categorical": False,
                    "lower": lower,
                    "upper": upper
                })
                
        return ps