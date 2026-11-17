# Non-blind deconvolution (kernel known): Wiener and Richardson-Lucy
import numpy as np

from .degrade import blur

RL_CAP = 10.0   # upper bound on RL pixel values (true intensities are in [0, 1])


def _psf2otf(k, shape):
    big = np.zeros(shape)
    kh, kw = k.shape
    big[:kh, :kw] = k
    big = np.roll(big, (-(kh // 2), -(kw // 2)), axis=(0, 1))
    return np.fft.fft2(big)


def wiener(y, k, nsr):
    # nsr = noise-to-signal power ratio
    p = k.shape[0]
    yp = np.pad(y, p, mode="reflect")
    H = _psf2otf(k, yp.shape)
    X = np.fft.fft2(yp) * np.conj(H) / (np.abs(H) ** 2 + nsr)
    x = np.real(np.fft.ifft2(X))
    return x[p:-p, p:-p]


def richardson_lucy(y, k, n_iter, checkpoints=None):
    # With checkpoints (iteration counts) returns {iteration: image}, otherwise the final image
    y = np.maximum(y, 1e-6)
    kf = k[::-1, ::-1]
    x = y.copy()
    keep = set(checkpoints) if checkpoints is not None else None
    out = {}
    for i in range(1, n_iter + 1):
        est = np.maximum(blur(x, k), 1e-6)
        # a wrong kernel can make RL diverge to inf
        x = np.minimum(x * blur(y / est, kf), RL_CAP)
        if keep is not None and i in keep:
            out[i] = x.copy()
    return out if keep is not None else x


##################### sparse-gradient prior (step 8b) ###################################
HQS_BETA0, HQS_BETA_MAX, HQS_RATE = 1.0, 256.0, 2 * np.sqrt(2)
_LUTS = {}


def _shrink(v, alpha, beta):
    # argmin_w |w|^alpha + beta/2 (w - v)^2, elementwise
    if alpha == 1.0:
        return np.sign(v) * np.maximum(np.abs(v) - 1.0 / beta, 0.0)
    key = (round(alpha, 4), round(beta, 4))
    if key not in _LUTS:
        vg = np.linspace(-1.5, 1.5, 3001)
        t = np.linspace(0.0, 1.0, 501)[None, :]          # candidates w = t * v (minimizer lies in [0, v])
        cand = t * vg[:, None]
        cost = np.abs(cand) ** alpha + beta / 2 * (cand - vg[:, None]) ** 2
        _LUTS[key] = (vg, cand[np.arange(len(vg)), cost.argmin(1)])
    vg, wg = _LUTS[key]
    return np.interp(v, vg, wg)


def prior_deconv(y, k, lam, alpha=2 / 3):
    # Non-blind deconvolution with a sparse-gradient prior, half-quadratic splitting:
    # min_x  lam/2 ||k * x - y||^2 + sum |grad x|^alpha
    p = k.shape[0]
    yp = np.pad(y, p, mode="reflect")
    sh = yp.shape
    K = _psf2otf(k, sh)
    dx, dy = np.zeros(sh), np.zeros(sh)
    dx[0, 0], dx[0, 1], dy[0, 0], dy[1, 0] = -1, 1, -1, 1
    Dx, Dy = np.fft.fft2(dx), np.fft.fft2(dy)
    num = np.conj(K) * np.fft.fft2(yp)
    den_k, den_d = np.abs(K) ** 2, np.abs(Dx) ** 2 + np.abs(Dy) ** 2
    x, beta = yp.copy(), HQS_BETA0
    while beta < HQS_BETA_MAX:
        Xf = np.fft.fft2(x)
        wx = _shrink(np.real(np.fft.ifft2(Xf * Dx)), alpha, beta)
        wy = _shrink(np.real(np.fft.ifft2(Xf * Dy)), alpha, beta)
        g = beta / lam
        x = np.real(np.fft.ifft2((num + g * (np.conj(Dx) * np.fft.fft2(wx) + np.conj(Dy) * np.fft.fft2(wy))) / (den_k + g * den_d)))
        beta *= HQS_RATE
    return x[p:-p, p:-p]
