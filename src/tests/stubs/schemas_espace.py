"""STUB temporaire : seulement divergence() (centrée), pour max_divergence."""
import numpy as np
def divergence(u, v, dx, dy):
    d = np.zeros_like(u)
    d[1:-1, 1:-1] = ((u[2:, 1:-1] - u[:-2, 1:-1]) / (2 * dx)
                     + (v[1:-1, 2:] - v[1:-1, :-2]) / (2 * dy))
    return d
