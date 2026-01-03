import numpy as np

def reconstruction_error(x: np.ndarray, x_hat: np.ndarray) -> float:
    """Normalized reconstruction error. 0 is perfect reconstruction."""
    denom = np.linalg.norm(x)
    if denom == 0:
        return 0.0
    return float(np.linalg.norm(x - x_hat) / denom)

def coherence_score(x: np.ndarray, x_hat: np.ndarray) -> float:
    """Coherence score in [0, 1]. Higher is better."""
    err = reconstruction_error(x, x_hat)
    return max(0.0, 1.0 - err)
