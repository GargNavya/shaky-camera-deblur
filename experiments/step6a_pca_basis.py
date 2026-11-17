# Step 6a: learn the PCA kernel basis and measure how well it can represent each kernel family
import numpy as np

from experiments.step6_common import family_kernels, phase_dir
from deblur.kernels import kernel_similarity
from deblur.pca import KernelBasis, training_kernels
from deblur.plotting import plt
from deblur.tables import write_csv


def run(cfg, root, workers, log):
    out = phase_dir(root, "6a")
    n_max = max(cfg["pca_train_sizes"])
    log.info(f"generating {n_max} training kernels")
    train = training_kernels(n_max)
    bases = {}
    for n in cfg["pca_train_sizes"]:
        for aligned in (True, False):
            bases[(n, aligned)] = KernelBasis.fit(train[:n], aligned, cfg["pca_kmax"])
            log.info(f"  basis n={n} aligned={aligned} done")
    n_main = cfg["pca_train_n"]
    bases[(n_main, True)].save(out / "basis_aligned.npz")
    bases[(n_main, False)].save(out / "basis_unaligned.npz")

    # explained variance (relative to the total variance of the training set)
    rows = []
    for aligned in (True, False):
        b = bases[(n_main, aligned)]
        ev = b.explained_variance()
        for K in range(1, len(ev) + 1):
            rows.append(dict(aligned=aligned, K=K, explained_variance=ev[K - 1]))
    write_csv(out / "explained_variance.csv", rows)
    fig, ax = plt.subplots(figsize=(5, 3.6))
    for aligned, col in ((True, "#08519c"), (False, "#9ecae1")):
        r = [x for x in rows if x["aligned"] == aligned]
        ax.plot([x["K"] for x in r], [x["explained_variance"] for x in r], color=col, lw=2,
                label="rotated to shake direction" if aligned else "not rotated")
    ax.set_xlabel("number of components K")
    ax.set_ylabel("fraction of variance (of first 60)")
    ax.set_title("PCA of curved training kernels", fontsize=10, loc="left")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "explained_variance.png", dpi=110)
    plt.close(fig)

    # capacity: similarity of the best K-term approximation to each test kernel
    kernels = family_kernels(cfg["test_kernels"], "test")
    cap = []
    for (n, aligned), b in bases.items():
        for fam, kid, k in kernels:
            for K in cfg["capacity_K"]:
                if K <= b.comps.shape[0]:
                    cap.append(dict(train_n=n, aligned=aligned, family=fam, kernel_id=kid, K=K,
                                    similarity=kernel_similarity(b.project(k, K), k)))
    write_csv(out / "capacity.csv", cap)
    fams = [f for f in ("straight", "curved", "traj", "real") if any(c["family"] == f for c in cap)]
    fig, axes = plt.subplots(1, len(fams), figsize=(4 * len(fams), 3.5), squeeze=False)
    for ax, fam in zip(axes[0], fams):
        for (n, aligned), b in bases.items():
            if n != n_main and not aligned:
                continue
            Ks = cfg["capacity_K"]
            v = [np.mean([c["similarity"] for c in cap if c["train_n"] == n and c["aligned"] == aligned
                          and c["family"] == fam and c["K"] == K]) for K in Ks if K <= b.comps.shape[0]]
            ls = "-" if aligned else "--"
            col = {True: "#08519c", False: "#9ecae1"}[aligned] if n == n_main else "#bdbdbd"
            ax.plot(Ks[:len(v)], v, ls=ls, color=col, marker="o", ms=3,
                    label=f"{'rotated' if aligned else 'not rotated'}, n={n}")
        ax.set_xscale("log")
        ax.set_xticks(cfg["capacity_K"])
        ax.set_xticklabels([str(k) for k in cfg["capacity_K"]], fontsize=8)
        ax.minorticks_off()
        ax.set_title(f"{fam} kernels", fontsize=10, loc="left")
        ax.set_xlabel("K")
        ax.set_ylabel("similarity of projection to truth")
        ax.grid(alpha=0.25)
    axes[0][0].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "capacity.png", dpi=110)
    plt.close(fig)

    b = bases[(n_main, True)]
    m, B = b.frame(min(9, b.comps.shape[0]), 0.0)
    fig, axes = plt.subplots(1, len(B) + 1, figsize=(1.6 * (len(B) + 1), 1.9))
    for ax, img, t in zip(axes, [m] + B, ["mean"] + [f"PC{i + 1}" for i in range(len(B))]):
        v = np.abs(img).max()
        ax.imshow(img, cmap="RdBu_r" if t != "mean" else "gray", vmin=-v if t != "mean" else 0, vmax=v)
        ax.set_title(t, fontsize=8)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out / "basis_images.png", dpi=110)
    plt.close(fig)

    for fam in fams:
        for aligned in (True, False):
            msg = " ".join(f"K={K}:{np.mean([c['similarity'] for c in cap if c['train_n'] == n_main and c['aligned'] == aligned and c['family'] == fam and c['K'] == K]):.3f}"
                           for K in cfg["capacity_K"] if K <= cfg["pca_kmax"])
            log.info(f"capacity {fam:8s} {'rotated' if aligned else 'not rotated':11s} {msg}")
    ev = bases[(n_main, True)].explained_variance()
    log.info(f"explained variance (rotated) K=5/10/20: {ev[4]:.3f}/{ev[9]:.3f}/{ev[min(19, len(ev) - 1)]:.3f}")
    return f"bases saved, {len(cap)} capacity rows"
