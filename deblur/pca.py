import numpy as np

from .config import KERNEL_SIZE
from .kernels import align_kernel, center_kernel, curved_kernel, kernel_angle_deg, normalize_kernel, rotate_kernel

TRAIN_SEED = 1_000_003
TRAIN_LENGTH = (5.0, 25.0)
TRAIN_BEND = (0.0, 150.0)


def training_kernels(n, rng=None, ksize=KERNEL_SIZE):
    # n random curved kernels
    rng = rng or np.random.default_rng(TRAIN_SEED)
    out = []
    while len(out) < n:
        try:
            out.append(curved_kernel(rng.uniform(*TRAIN_LENGTH), rng.uniform(*TRAIN_BEND), rng, ksize))
        except ValueError:
            pass
    return out


class KernelBasis:
    def __init__(self, mean, comps, sv, aligned):
        self.mean, self.comps, self.sv, self.aligned = mean, comps, sv, bool(aligned)
        self.ksize = int(round(np.sqrt(mean.size)))
        self._cache = {}

    @classmethod
    def fit(cls, kernels, aligned, kmax=60):
        prep = align_kernel if aligned else center_kernel
        X = np.stack([prep(k).ravel() for k in kernels])
        mean = X.mean(axis=0)
        _, s, vt = np.linalg.svd(X - mean, full_matrices=False)
        return cls(mean, vt[:kmax], s[:kmax] / np.sqrt(len(X)), aligned)

    def explained_variance(self, total=None):
        v = self.sv ** 2
        return np.cumsum(v) / (total if total is not None else v.sum())

    def save(self, path):
        np.savez(path, mean=self.mean, comps=self.comps, sv=self.sv, aligned=self.aligned)

    @classmethod
    def load(cls, path):
        d = np.load(path)
        return cls(d["mean"], d["comps"], d["sv"], bool(d["aligned"]))

    def frame(self, K, theta=0.0):
        th = round(float(theta), 1) if self.aligned else 0.0
        key = (K, th)
        if key not in self._cache:
            ks = self.ksize
            m = self.mean.reshape(ks, ks)
            B = [b.reshape(ks, ks) for b in self.comps[:K]]
            if self.aligned and th != 0.0:
                from scipy.ndimage import rotate
                m = rotate(m, th, reshape=False, order=1)
                B = [rotate(b, th, reshape=False, order=1) for b in B]
            if len(self._cache) > 64:
                self._cache.clear()
            self._cache[key] = (m, B)
        return self._cache[key]

    def project(self, k, K, theta=None):
        theta = kernel_angle_deg(k) if theta is None else theta
        a = (align_kernel(k) if self.aligned else center_kernel(k)).ravel()
        c = (a - self.mean) @ self.comps[:K].T
        rec = (self.mean + c @ self.comps[:K]).reshape(self.ksize, self.ksize)
        rec = normalize_kernel(rec)
        return rotate_kernel(rec, theta) if self.aligned else rec
