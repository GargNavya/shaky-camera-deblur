import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def psnr(ref, est):
    return float(peak_signal_noise_ratio(ref, np.clip(est, 0, 1), data_range=1.0))

def ssim(ref, est):
    return float(structural_similarity(ref, np.clip(est, 0, 1), data_range=1.0))

def mse_aligned(x, xh, max_shift=3, border=20):
    xh = np.clip(xh, 0, 1)
    b = border
    ref = x[b:-b, b:-b]
    best = np.inf
    for dy in range(-max_shift, max_shift + 1):
        for dx in range(-max_shift, max_shift + 1):
            cand = xh[b + dy:x.shape[0] - b + dy, b + dx:x.shape[1] - b + dx]
            best = min(best, float(np.mean((ref - cand) ** 2)))
    return best

def psnr_from_mse(m):
    return float(10 * np.log10(1.0 / max(m, 1e-12)))

def best_shift(x, xh, max_shift=3, border=20):
    xh = np.clip(xh, 0, 1)
    b = border
    ref = x[b:-b, b:-b]
    best, arg = np.inf, (0, 0)
    for dy in range(-max_shift, max_shift + 1):
        for dx in range(-max_shift, max_shift + 1):
            m = float(np.mean((ref - xh[b + dy:x.shape[0] - b + dy, b + dx:x.shape[1] - b + dx]) ** 2))
            if m < best:
                best, arg = m, (dy, dx)
    return arg

def region_masks(x, flat_frac=0.5, edge_frac=0.1, border=20):
    from scipy.ndimage import gaussian_filter
    gy, gx = np.gradient(gaussian_filter(x, 1.0))
    g = gaussian_filter(np.hypot(gx, gy), 2.0)[border:-border, border:-border]
    return g <= np.quantile(g, flat_frac), g >= np.quantile(g, 1 - edge_frac)

def psnr_regions(x, xh, masks=None, border=20):
    dy, dx = best_shift(x, xh, border=border)
    b = border
    err = (x[b:-b, b:-b] - np.clip(xh, 0, 1)[b + dy:x.shape[0] - b + dy, b + dx:x.shape[1] - b + dx]) ** 2
    flat, edge = masks if masks is not None else region_masks(x, border=border)
    p = lambda m: float(10 * np.log10(1.0 / max(m, 1e-12)))
    from skimage.metrics import structural_similarity
    al = np.clip(xh, 0, 1)[b + dy:x.shape[0] - b + dy, b + dx:x.shape[1] - b + dx]
    return dict(psnr=p(err.mean()), psnr_flat=p(err[flat].mean()), psnr_edge=p(err[edge].mean()),
                ssim=float(structural_similarity(x[b:-b, b:-b], al, data_range=1.0)))
