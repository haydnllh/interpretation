import numpy as np

def precision_coverage(
    X,
    X_samples,
    anchor,
    model,
    is_prob
):
    """Computes the precision and coverage for an anchor"""
    pred_X = model(X.reshape(1, -1))
    pred_samples = model(X_samples)
    mask = np.ones(X_samples.shape[0], dtype=bool)
    
    binary = (pred_samples.ndim == 2 and pred_samples.shape[-1] == 1) \
        or (pred_samples.ndim == 1)
    
    if is_prob and binary:
        label_X = pred_X > 0.5
        label_samples = (pred_samples > 0.5).astype(int)
    elif is_prob and not binary:
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