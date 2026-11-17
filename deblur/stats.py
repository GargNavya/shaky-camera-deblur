# Paired statistics for method comparisons
import numpy as np
from scipy.stats import wilcoxon

N_BOOT = 10_000


def cluster_bootstrap_ci(diff, clusters, n_boot=N_BOOT, alpha=0.05, seed=0):
    diff, clusters = np.asarray(diff, float), np.asarray(clusters)
    ids = np.unique(clusters)
    groups = [diff[clusters == i] for i in ids]
    sums = np.array([g.sum() for g in groups])
    counts = np.array([len(g) for g in groups])
    rng = np.random.default_rng(seed)
    pick = rng.integers(0, len(ids), size=(n_boot, len(ids)))
    boot = sums[pick].sum(1) / counts[pick].sum(1)
    lo, hi = np.quantile(boot, [alpha / 2, 1 - alpha / 2])
    return float(diff.mean()), float(lo), float(hi)


def paired_test(a, b, clusters, alternative="two-sided", margin=0.0):
    a, b, clusters = np.asarray(a, float), np.asarray(b, float), np.asarray(clusters)
    d = a - b
    mean, lo, hi = cluster_bootstrap_ci(d, clusters)
    dm = d + margin
    per_img = np.array([dm[clusters == i].mean() for i in np.unique(clusters)])
    p_img = wilcoxon(per_img, alternative=alternative).pvalue if np.any(per_img != 0) and len(per_img) > 1 else 1.0
    p_case = wilcoxon(dm, alternative=alternative).pvalue if np.any(dm != 0) else 1.0
    return dict(mean_diff=mean, ci_lo=lo, ci_hi=hi, win_rate=float(np.mean(d > 0)), ties=float(np.mean(d == 0)), n_cases=len(d), n_images=len(per_img), p_wilcoxon_images=float(p_img), p_wilcoxon_cases=float(p_case))


def holm(pvals):
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(p) - rank) * p[i]))
        adj[i] = running
    return adj
