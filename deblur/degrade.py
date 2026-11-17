# Forward model: y = k * x + n
# Reflect padding at borders
# same-size output
import numpy as np
from scipy.signal import fftconvolve


def blur(img, kernel):
    # Convolve image with kernel
    p = kernel.shape[0] // 2
    padded = np.pad(img, p, mode="reflect")
    return fftconvolve(padded, kernel, mode="valid")


def add_noise(img, sigma, rng, clip=False):
    # Gaussian noise, std sigma on the [0, 1] intensity scale
    out = img + rng.normal(0.0, sigma, size=img.shape)
    return np.clip(out, 0, 1) if clip else out


def make_blurred(img, kernel, sigma, rng):
    return add_noise(blur(img, kernel), sigma, rng)
