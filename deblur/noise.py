import numpy as np

def estimate_sigma(y):
    h, w = (y.shape[0] // 2) * 2, (y.shape[1] // 2) * 2
    y = y[:h, :w]
    hh = (y[0::2, 0::2] - y[0::2, 1::2] - y[1::2, 0::2] + y[1::2, 1::2]) / 2.0
    return float(np.median(np.abs(hh)) / 0.6745)

def nearest_level(sigma, levels):
    levels = sorted(levels)
    return min(levels, key=lambda s: abs(s - sigma))
