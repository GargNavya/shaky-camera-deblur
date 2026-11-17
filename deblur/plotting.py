"""Shared figure helpers. Experiments compute; this module only draws."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# one-hue ramp, light to dark, for an ordered quantity such as noise level
BLUES = ["#c6dbef", "#9ecae1", "#6baed6", "#3182bd", "#08519c"]


def image_grid(rows, path, cell=3.4, dpi=100, title_size=8):
    """Grid of grayscale images.

    rows: list of rows; each row is a list of (image, title, kernel_or_None).
    A kernel is drawn as a small inset in the lower right corner.
    """
    nr, nc = len(rows), max(len(r) for r in rows)
    fig, axes = plt.subplots(nr, nc, figsize=(cell * nc, cell * nr), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for r, row in enumerate(rows):
        for c, (img, title, kern) in enumerate(row):
            ax = axes[r, c]
            ax.imshow(np.clip(img, 0, 1), cmap="gray", vmin=0, vmax=1)
            ax.set_title(title, fontsize=title_size)
            if kern is not None:
                ins = ax.inset_axes([0.70, 0.02, 0.28, 0.28])
                ins.imshow(kern / kern.max(), cmap="gray")
                ins.set_xticks([])
                ins.set_yticks([])
    plt.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def chunk(items, n):
    """Split a list into rows of n."""
    return [items[i:i + n] for i in range(0, len(items), n)]


def ablation_figure(sigmas, nsrs, wiener_psnr, iters, rl_psnr, blurred_psnr, path):
    """Two panels: Wiener PSNR vs nsr, RL PSNR vs iterations; one line per noise level.

    wiener_psnr: [sigma, nsr]   rl_psnr: [sigma, iter]   blurred_psnr: [sigma]
    All are means over images. Dotted lines: PSNR of the blurred input.
    """
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for si, sigma in enumerate(sigmas):
        col = BLUES[si % len(BLUES)]
        axes[0].plot(nsrs, wiener_psnr[si], color=col, lw=2, marker="o", ms=4, label=f"sigma={sigma:g}")
        axes[1].plot(iters, rl_psnr[si], color=col, lw=2, marker="o", ms=4, label=f"sigma={sigma:g}")
        for ax in axes:
            ax.axhline(blurred_psnr[si], color=col, lw=1, ls=":")
    axes[0].set_xscale("log")
    axes[0].set_xlabel("Wiener noise-to-signal ratio (nsr)")
    axes[1].set_xlabel("Richardson-Lucy iterations")
    for ax, t in zip(axes, ["Wiener: PSNR vs nsr", "Richardson-Lucy: PSNR vs iterations"]):
        ax.set_title(t, fontsize=10, loc="left")
        ax.set_ylabel("mean PSNR (dB)")
        ax.grid(alpha=0.25, lw=0.6)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.legend(fontsize=8, frameon=False, title="dotted = blurred input", title_fontsize=7)
    plt.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _clean(ax):
    ax.grid(alpha=0.25, lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


STEP4_STYLE = {"structure tensor": ("#9ecae1", "--"), "cepstrum (free)": ("#3182bd", "-"),
               "cepstrum + ST hint": ("#08519c", "-")}


def method_curves(x, curves, xlabel, ylabels, titles, path, hline=None, style=None, xticklabels=None):
    """Panels side by side; each panel has one line per method.

    x: x values. curves: {panel_index: {method: y_values}} where panel i uses ylabels[i].
    style: {method: (color, linestyle)}; defaults to the step 4 methods.
    xticklabels: optional labels for a categorical x axis.
    """
    n = len(titles)
    style = style or STEP4_STYLE
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 3.8), squeeze=False)
    for i, ax in enumerate(axes[0]):
        for m, ys in curves[i].items():
            col, ls = style.get(m, ("#555555", "-"))
            ax.plot(x, ys, color=col, ls=ls, lw=2, marker="o", ms=4, label=m)
        if hline is not None and i in hline:
            ax.axhline(hline[i], color="#999", lw=1, ls=":")
        if xticklabels is not None:
            ax.set_xticks(list(x))
            ax.set_xticklabels(xticklabels, fontsize=8)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabels[i])
        ax.set_title(titles[i], fontsize=10, loc="left")
        _clean(ax)
    axes[0][0].legend(fontsize=8, frameon=False)
    plt.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def cepstrum_figure(panels, path):
    """panels: list of (cepstrum_array, true_theta, est_theta, true_L, est_L, title)."""
    fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 4.2), squeeze=False)
    for ax, (C, tt, et, tl, el, title) in zip(axes[0], panels):
        cy, cx = C.shape[0] // 2, C.shape[1] // 2
        w = 36
        sub = C[cy - w:cy + w + 1, cx - w:cx + w + 1]
        v = np.abs(sub).max()
        ax.imshow(sub, cmap="Blues_r", vmin=-v, vmax=v, extent=[-w, w, w, -w])
        for th, L, mk, lab in [(tt, tl, "o", "true"), (et, el, "x", "estimate")]:
            ax.plot(L * np.cos(np.deg2rad(th)), -L * np.sin(np.deg2rad(th)), mk, color="#d94801",
                    mfc="none", ms=11, mew=2, label=lab)
        ax.set_title(title, fontsize=9, loc="left")
        ax.legend(fontsize=7, frameon=False, loc="lower right")
    plt.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def draw_field(ax, img, field, color="#ffb000", scale=1.0, title="", center=None):
    """Blur vectors of a kernel field drawn as line segments (true length x scale) over an image.
    field: dict with theta (deg), L (px), centers (ny, nx, 2) in (x, y)."""
    ax.imshow(img, cmap="gray", vmin=0, vmax=1)
    C = field["centers"].reshape(-1, 2)
    t = np.deg2rad(field["theta"].ravel())
    L = field["L"].ravel() * scale / 2
    for (cx, cy), a, l in zip(C, t, L):
        ax.plot([cx - l * np.cos(a), cx + l * np.cos(a)], [cy + l * np.sin(a), cy - l * np.sin(a)], color=color, lw=1.6)
    if center is not None and np.all(np.isfinite(center)):
        h, w = img.shape[:2]
        if -0.5 * w <= center[0] <= 1.5 * w and -0.5 * h <= center[1] <= 1.5 * h:
            ax.plot(center[0], center[1], "x", color="#d94801", ms=10, mew=2)
    ax.set_xlim(-0.5, img.shape[1] - 0.5)
    ax.set_ylim(img.shape[0] - 0.5, -0.5)
    ax.set_title(title, fontsize=8, loc="left")
    ax.axis("off")


def draw_arcs(ax, img, center, total_angle, n=7, color="#08519c", title=""):
    """Circle model drawn over an image: on an n x n grid of points, the arc each point travels
    (radius = distance to the epicenter, angle = total_angle), with the epicenter marked."""
    ax.imshow(img, cmap="gray", vmin=0, vmax=1)
    h, w = img.shape[:2]
    cx, cy = center
    for py in np.linspace(h * 0.08, h * 0.92, n):
        for px in np.linspace(w * 0.08, w * 0.92, n):
            r = np.hypot(px - cx, py - cy)
            a0 = np.arctan2(py - cy, px - cx)
            a = a0 + np.linspace(-total_angle / 2, total_angle / 2, 20)
            ax.plot(cx + r * np.cos(a), cy + r * np.sin(a), color=color, lw=1.6)
    if -0.5 * w <= cx <= 1.5 * w and -0.5 * h <= cy <= 1.5 * h:
        ax.plot(cx, cy, "x", color="#d94801", ms=10, mew=2)
    ax.set_xlim(-0.5, w - 0.5)
    ax.set_ylim(h - 0.5, -0.5)
    ax.set_title(title, fontsize=8, loc="left")
    ax.axis("off")
