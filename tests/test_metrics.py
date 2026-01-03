import numpy as np
from cohbench.metrics import reconstruction_error, coherence_score

def test_perfect_reconstruction():
    x = np.array([1.0, 2.0, 3.0])
    assert reconstruction_error(x, x) == 0.0
    assert coherence_score(x, x) == 1.0

def test_imperfect_reconstruction():
    x = np.array([1.0, 2.0, 3.0])
    x_hat = np.array([1.0, 2.5, 2.0])
    score = coherence_score(x, x_hat)
    assert 0.0 <= score < 1.0
