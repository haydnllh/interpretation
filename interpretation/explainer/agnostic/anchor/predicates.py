import numpy as np

def generate_predicates(
    X,
    categorical_features,
    lower_q,
    median,
    upper_q
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
            lq, m, uq = lower_q[i], median[i], upper_q[i]
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