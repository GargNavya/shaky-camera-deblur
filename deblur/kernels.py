# Conventions:-
# Kernel: odd square array, non-negative, sum 1, centered at (ksize//2, ksize//2)
# theta in degrees: 0 = horizontal, 90 = vertical, counter-clockwise positive on screen
import numpy as np
from scipy.ndimage import gaussian_filter1d

from .config import KERNEL_SIZE


def _splat(points_xy, weights, ksize):
    # Bilinear-splat weighted points (x = col, y = row, kernel coordinates) into a kernel
    k = np.zeros((ksize, ksize))
    x0 = np.floor(points_xy[:, 0]).astype(int)
    y0 = np.floor(points_xy[:, 1]).astype(int)
    fx = points_xy[:, 0] - x0
    fy = points_xy[:, 1] - y0
    for dx, dy, w in [
        (0, 0, (1 - fx) * (1 - fy)),
        (1, 0, fx * (1 - fy)),
        (0, 1, (1 - fx) * fy),
        (1, 1, fx * fy),
    ]:
        xi, yi = x0 + dx, y0 + dy
        ok = (xi >= 0) & (xi < ksize) & (yi >= 0) & (yi < ksize)
        np.add.at(k, (yi[ok], xi[ok]), (w * weights)[ok])
    s = k.sum()
    if s <= 0:
        raise ValueError("empty kernel (path left the kernel window)")
    return k / s


def line_kernel(theta_deg, length, ksize=KERNEL_SIZE):
    # Straight motion-blur kernel
    # length in pixels
    assert ksize % 2 == 1
    c = ksize // 2
    length = max(float(length), 1.0)
    if length / 2 > c - 1:
        raise ValueError(f"length {length} too large for ksize {ksize}")
    th = np.deg2rad(theta_deg)
    n = int(length * 8) + 1
    t = np.linspace(-length / 2, length / 2, n)
    pts = np.stack([c + t * np.cos(th), c - t * np.sin(th)], axis=1)
    return _splat(pts, np.ones(n), ksize)


def curved_kernel(length, bend_deg, rng, ksize=KERNEL_SIZE, theta_deg=None):
    # Random smooth curved camera-shake kernel
    # Params:
    # length    approximate path length in pixels
    # bend_deg  how far the heading can turn along the path (0 = straight)
    # theta_deg starting heading (random if None)
    assert ksize % 2 == 1
    c = ksize // 2
    n = int(max(length, 1.0) * 8) + 1
    th0 = rng.uniform(0, 2 * np.pi) if theta_deg is None else np.deg2rad(theta_deg)
    s = gaussian_filter1d(rng.normal(size=n), n / 5, mode="nearest")
    s = s / (np.abs(s).max() + 1e-12)
    heading = th0 + np.deg2rad(bend_deg) * s
    step = length / max(n - 1, 1)
    xy = np.cumsum(step * np.stack([np.cos(heading), -np.sin(heading)], axis=1), axis=0)
    xy -= xy.mean(axis=0)
    reach = np.abs(xy).max()
    if reach > c - 1.5:
        xy *= (c - 1.5) / reach
    # camera dwells longer in some places: smooth random weights
    w = np.exp(gaussian_filter1d(rng.normal(size=n), n / 8, mode="nearest"))
    return _splat(xy + c, w, ksize)


def _kernel_cov(k):
    c = k.shape[0] // 2
    ys, xs = np.mgrid[0:k.shape[0], 0:k.shape[1]]
    x = xs - c
    y = -(ys - c)
    mx, my = (k * x).sum(), (k * y).sum()
    cxx = (k * (x - mx) ** 2).sum()
    cyy = (k * (y - my) ** 2).sum()
    cxy = (k * (x - mx) * (y - my)).sum()
    return cxx, cyy, cxy


def kernel_angle_deg(k):
    cxx, cyy, cxy = _kernel_cov(k)
    return float(np.rad2deg(0.5 * np.arctan2(2 * cxy, cxx - cyy)) % 180)


def kernel_anisotropy(k):
    # lambda_min / lambda_max of the kernel covariance
    cxx, cyy, cxy = _kernel_cov(k)
    gap = np.sqrt(((cxx - cyy) / 2) ** 2 + cxy ** 2)
    lmax, lmin = (cxx + cyy) / 2 + gap, (cxx + cyy) / 2 - gap
    return float(max(lmin, 0.0) / lmax)


def trajectory_kernel(length, rng, ksize=KERNEL_SIZE, n_steps=300):
    c = ksize // 2
    anxiety = rng.uniform(0.03, 0.15)
    p_jerk = rng.uniform(0.0, 0.01)
    centripetal = rng.uniform(0.0, 0.0005)
    v = np.exp(1j * rng.uniform(0, 2 * np.pi))
    x = np.zeros(n_steps, dtype=complex)
    for t in range(1, n_steps):
        dv = anxiety * (rng.normal() + 1j * rng.normal()) - centripetal * x[t - 1]
        if rng.uniform() < p_jerk:
            dv += rng.uniform(0.5, 1.5) * np.exp(1j * rng.uniform(0, 2 * np.pi))
        v = v + dv
        v = v / (abs(v) + 1e-12)
        x[t] = x[t - 1] + v
    seg = np.abs(np.diff(x)).sum()
    x = (x - x.mean()) * (length / max(seg, 1e-9))
    pts = np.stack([x.real, -x.imag], axis=1)
    reach = np.abs(pts).max()
    if reach > c - 1.5:
        pts *= (c - 1.5) / reach
    return _splat(pts + c, np.ones(n_steps), ksize)


def normalize_kernel(k):
    k = np.maximum(k, 0.0)
    s = k.sum()
    if s <= 0:
        raise ValueError("empty kernel")
    return k / s


def center_kernel(k):
    from scipy.ndimage import shift as nd_shift
    c = k.shape[0] // 2
    ys, xs = np.mgrid[0:k.shape[0], 0:k.shape[1]]
    s = k.sum()
    cy, cx = (k * ys).sum() / s, (k * xs).sum() / s
    return normalize_kernel(nd_shift(k, (c - cy, c - cx), order=1))


def rotate_kernel(k, deg):
    from scipy.ndimage import rotate
    return normalize_kernel(rotate(k, deg, reshape=False, order=1))


def align_kernel(k):
    k = center_kernel(k)
    return center_kernel(rotate_kernel(k, -kernel_angle_deg(k)))


def kernel_similarity(a, b):
    from scipy.signal import fftconvolve
    cc = fftconvolve(a, b[::-1, ::-1], mode="full")
    return float(cc.max() / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
