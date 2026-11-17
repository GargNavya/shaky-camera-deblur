import numpy as np
from scipy.ndimage import map_coordinates

CEP_EPS = 1e-3       # log floor, relative to mean spectral power
CEP_KAPPA = 4.0      # keep frequencies where radial signal power >= kappa * noise floor
CEP_FC_MIN = 0.04    # can't use a cutoff below this (cycles/px)
CEP_RMIN = 3.0       # smallest frequency searched (px); below this image content dominates
CEP_RMAX = 30.0      # largest frequency searched (px)
HINT_TOL = 20.0      # degrees searched around theta_hint
REFINE_TOL = 10.0    # degrees: angle refinement window around the first angle
REFINE_DR = 0.5      # px: radius band used for the refinement
LEN_TOL = 5.0        # degrees around the chosen angle used to read off the length


def structure_tensor_angle(y):
    # Direction of least gradient energy
    # Returns (theta_deg, anisotropy = lambda_min / lambda_max)
    gy, gx = np.gradient(y)
    gy = -gy
    jxx, jyy, jxy = np.sum(gx * gx), np.sum(gy * gy), np.sum(gx * gy)
    phi_max = 0.5 * np.arctan2(2 * jxy, jxx - jyy)   # axis of most gradient energy
    theta = (np.rad2deg(phi_max) + 90.0) % 180.0
    half_gap = np.sqrt(((jxx - jyy) / 2) ** 2 + jxy ** 2)
    lam_max, lam_min = (jxx + jyy) / 2 + half_gap, (jxx + jyy) / 2 - half_gap
    return float(theta), float(lam_min / lam_max)


def _radial_bins(shape):
    h, w = shape
    fr = np.hypot(np.fft.fftfreq(h)[:, None], np.fft.fftfreq(w)[None, :])
    return fr, np.round(fr * h).astype(int)


def _radial_mean(a, idx):
    return np.bincount(idx.ravel(), a.ravel()) / np.maximum(np.bincount(idx.ravel()), 1)


def signal_cutoff(P, fr, idx, kappa=CEP_KAPPA):
    n0 = np.median(P[fr > 0.4]) / np.log(2.0)
    prof = np.convolve(_radial_mean(P, idx), np.ones(5) / 5, mode="same")
    h = P.shape[0]
    below = np.where(prof[3: h // 2] <= kappa * n0)[0]
    fc = (below[0] + 3) / h if below.size else 0.5
    return float(np.clip(fc, CEP_FC_MIN, 0.5))


def cepstrum(y, kappa=CEP_KAPPA, eps=CEP_EPS):
    h, w = y.shape
    win = np.outer(np.hanning(h), np.hanning(w))
    P = np.abs(np.fft.fft2((y - y.mean()) * win)) ** 2
    fr, idx = _radial_bins(P.shape)
    fc = signal_cutoff(P, fr, idx, kappa)
    L = np.log(P + eps * P.mean())
    L = L - L[fr < fc].mean()
    taper = np.clip((fc - fr) / (0.25 * fc), 0.0, 1.0)   # 1 below 0.75 fc, 0 at fc
    return np.fft.fftshift(np.real(np.fft.ifft2(L * taper))), fc


def _polar(C, rmin, rmax, dr=0.25, dtheta=1.0):
    # Sample the cepstrum on an (angle, radius) grid; subtract the mean over angle
    cy, cx = C.shape[0] // 2, C.shape[1] // 2
    thetas = np.arange(0.0, 180.0, dtheta)
    rs = np.arange(rmin, rmax + 1e-9, dr)
    T, R = np.meshgrid(np.deg2rad(thetas), rs, indexing="ij")
    v = map_coordinates(C, [cy - R * np.sin(T), cx + R * np.cos(T)], order=1)
    return thetas, rs, v - v.mean(axis=0, keepdims=True)


def _within(thetas, center, tol):
    d = np.abs(thetas - center) % 180.0
    return np.minimum(d, 180.0 - d) <= tol


def cepstrum_estimate(y, theta_hint=None, kappa=CEP_KAPPA, rmin=CEP_RMIN, rmax=CEP_RMAX, cep=None):
    # cep: optional precomputed cepstrum(y) result, to avoid recomputing
    C, fc = cep if cep is not None else cepstrum(y, kappa)
    thetas, rs, dev = _polar(C, rmin, rmax)
    z = dev / (dev.std(axis=0, keepdims=True) + 1e-12)      # unit spread at every radius
    if theta_hint is not None:
        z = np.where(_within(thetas, theta_hint, HINT_TOL)[:, None], z, np.inf)
    i, _ = np.unravel_index(np.argmin(z), z.shape)
    theta = thetas[i]
    raw = np.where(_within(thetas, theta, LEN_TOL)[:, None], dev, np.inf)
    _, j = np.unravel_index(np.argmin(raw), raw.shape)
    # refine the angle at the chosen length: strongest dip near that radius, within +-REFINE_TOL
    near = np.abs(rs - rs[j]) <= REFINE_DR
    prof = np.where(_within(thetas, theta, REFINE_TOL), dev[:, near].mean(axis=1), np.inf)
    return float(thetas[np.argmin(prof)]), float(rs[j]), fc


def angle_error_deg(est, true):
    d = abs(est - true) % 180.0
    return float(min(d, 180.0 - d))


def estimate_line_kernel(y, ksize=None):
    # Blind straight-kernel estimate: cepstrum with structure-tensor hint
    from .config import KERNEL_SIZE
    ksize = ksize or KERNEL_SIZE
    th_st, _ = structure_tensor_angle(y)
    theta, length, _ = cepstrum_estimate(y, theta_hint=th_st)
    return theta, float(np.clip(length, 1.0, 2 * (ksize // 2 - 1)))


def estimate_line_kernel_conf(y, ksize=None):
    from .config import KERNEL_SIZE
    ksize = ksize or KERNEL_SIZE
    th_st, _ = structure_tensor_angle(y)
    C, fc = cepstrum(y)
    thetas, rs, dev = _polar(C, CEP_RMIN, CEP_RMAX)
    z = dev / (dev.std(axis=0, keepdims=True) + 1e-12)
    zh = np.where(_within(thetas, th_st, HINT_TOL)[:, None], z, np.inf)
    i, _ = np.unravel_index(np.argmin(zh), zh.shape)
    theta, length, _ = cepstrum_estimate(y, theta_hint=th_st, cep=(C, fc))
    j = int(np.argmin(np.abs(rs - length)))
    conf = float(-z[i, j])
    return theta, float(np.clip(length, 1.0, 2 * (ksize // 2 - 1))), conf
