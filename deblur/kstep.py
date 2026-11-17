# Kernel-estimation step: given the blurry image y and a sharp proxy x~, find the kernel k that best explains  grad(y) ~ k * grad(x~)  (least squares in the Fourier domain)
import numpy as np

from .config import KERNEL_SIZE
from .deconv import wiener
from .kernels import center_kernel, line_kernel, normalize_kernel

FREE_THRESH = 0.05
STRAIGHT_DTHETA = (4.0, 0.5)
STRAIGHT_DL = (2.5, 0.25)
TAPER = 32


def _grads(a):
    return [np.roll(a, -1, axis=1) - a, np.roll(a, -1, axis=0) - a]


def _taper(shape, w=TAPER):
    def ramp(n):
        r = np.ones(n)
        t = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, w))
        r[:w], r[-w:] = t, t[::-1]
        return r
    return np.outer(ramp(shape[0]), ramp(shape[1]))


def kernel_fft(k, shape):
    big = np.zeros(shape)
    ks = k.shape[0]
    big[:ks, :ks] = k
    return np.fft.fft2(np.roll(big, (-(ks // 2), -(ks // 2)), axis=(0, 1)))


class GradSystem:
    def __init__(self, y, proxy, keep=1.0):
        w = _taper(y.shape)
        gx = _grads(proxy)
        if keep < 1.0:
            mag = np.hypot(*gx)
            mask = mag >= np.quantile(mag, 1.0 - keep)
            gx = [g * mask for g in gx]
        GX = [np.fft.fft2(g * w) for g in gx]
        GY = [np.fft.fft2(g * w) for g in _grads(y)]
        self.shape = y.shape
        self.S = sum(np.abs(a) ** 2 for a in GX)
        self.T = sum(np.conj(a) * b for a, b in zip(GX, GY))
        self.Y2 = float(sum((np.abs(b) ** 2).sum() for b in GY))

    def residual(self, k):
        K = kernel_fft(k, self.shape)
        return float(self.Y2 - 2 * np.real((np.conj(K) * self.T).sum()) + (np.abs(K) ** 2 * self.S).sum())


def make_proxy(y, k, nsr):
    return wiener(y, k, nsr)


def step_free(sys, beta_rel, ksize=KERNEL_SIZE):
    Kf = sys.T / (sys.S + beta_rel * sys.S.mean())
    k = np.roll(np.real(np.fft.ifft2(Kf)), (ksize // 2, ksize // 2), axis=(0, 1))[:ksize, :ksize]
    k = np.maximum(k, 0)
    k[k < FREE_THRESH * k.max()] = 0
    return center_kernel(k)


def step_pca(sys, basis, K, theta, ridge_rel):
    m, B = basis.frame(K, theta)
    Mf = kernel_fft(m, sys.shape)
    Bf = np.stack([kernel_fft(b, sys.shape).ravel() for b in B])          # K x F
    S, T = sys.S.ravel(), sys.T.ravel()
    A = np.real((np.conj(Bf) * S) @ Bf.T)
    rhs = np.real(np.conj(Bf) @ (T - S * Mf.ravel()))
    var = basis.sv[:K] ** 2
    lam = ridge_rel * np.trace(A) / K
    c = np.linalg.solve(A + lam * np.diag(var[0] / var), rhs)
    k = m + np.tensordot(c, np.stack(B), axes=1)
    return center_kernel(normalize_kernel(k))


def step_straight(sys, theta0, length0, max_len=None, ksize=KERNEL_SIZE):
    max_len = max_len or 2 * (ksize // 2 - 1)
    best = (np.inf, theta0, length0)
    for th in theta0 + np.arange(-STRAIGHT_DTHETA[0], STRAIGHT_DTHETA[0] + 1e-9, STRAIGHT_DTHETA[1]):
        for L in length0 + np.arange(-STRAIGHT_DL[0], STRAIGHT_DL[0] + 1e-9, STRAIGHT_DL[1]):
            if 1.0 <= L <= max_len:
                r = sys.residual(line_kernel(th % 180.0, L))
                if r < best[0]:
                    best = (r, th % 180.0, L)
    return line_kernel(best[1], best[2]), best[1], best[2]


def kernel_step(model, sys, theta, length, cfg, basis=None, basis_unaligned=None):
    """One kernel update. Returns (kernel, theta, length); theta/length only change for 'straight'."""
    if model == "free":
        return step_free(sys, cfg["free_beta"]), theta, length
    if model == "pca":
        return step_pca(sys, basis, cfg["pca_K"], theta, cfg["pca_ridge"]), theta, length
    if model == "pca_unaligned":
        return step_pca(sys, basis_unaligned, cfg["pca_K"], 0.0, cfg["pca_ridge"]), theta, length
    if model == "straight":
        return step_straight(sys, theta, length)
    raise ValueError(model)
