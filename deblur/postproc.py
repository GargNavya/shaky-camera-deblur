import numpy as np
from scipy.ndimage import gaussian_filter, uniform_filter

from .deconv import _psf2otf

MASK_SMOOTH = 2.0   # px, Gaussian smoothing of the gradient magnitude for the flat-region mask


def notch_filter(x, k, eps):
    p = k.shape[0]
    xp = np.pad(x, p, mode="reflect")
    H2 = np.abs(_psf2otf(k, xp.shape)) ** 2
    return np.real(np.fft.ifft2(np.fft.fft2(xp) * H2 / (H2 + eps)))[p:-p, p:-p]


def flat_mask(y, tau):
    # 1 in flat regions, 0 at edges
    gy, gx = np.gradient(gaussian_filter(y, 1.0))
    g = gaussian_filter(np.hypot(gx, gy), MASK_SMOOTH)
    return np.exp(-(g / (tau * np.median(g) + 1e-12)) ** 2)


def mask_blend(x_sharp, x_gentle, y, tau):
    m = flat_mask(y, tau)
    return m * x_gentle + (1 - m) * x_sharp


def guided_smooth(x, r, eps):
    size = 2 * r + 1
    mean = uniform_filter(x, size, mode="reflect")
    var = uniform_filter(x * x, size, mode="reflect") - mean ** 2
    a = var / (var + eps)
    b = (1 - a) * mean
    return uniform_filter(a, size, mode="reflect") * x + uniform_filter(b, size, mode="reflect")
