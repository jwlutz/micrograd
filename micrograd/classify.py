"""Watch a micrograd MLP learn a 2-D decision boundary (moons / spiral).

`make_moons` / `make_spiral` generate toy datasets, `fit_classifier` trains an
MLP with a max-margin (hinge) loss, and `animate_classifier` renders the
decision boundary forming over training as a GIF.
"""

import io
import math
import random

import matplotlib
matplotlib.use("Agg")  # render to memory buffers, no display window
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from micrograd.engine import Value
from micrograd.nn import MLP


def make_moons(n=100, noise=0.10, seed=0):
    """Two interleaving half-circles, labels +1 / -1."""
    random.seed(seed)
    pts, labels = [], []
    half = n // 2
    for i in range(n):
        if i < half:
            t = math.pi * i / max(half - 1, 1)
            x, y, lab = math.cos(t), math.sin(t), 1
        else:
            j = i - half
            t = math.pi * j / max(n - half - 1, 1)
            x, y, lab = 1 - math.cos(t), 0.5 - math.sin(t), -1
        pts.append((x + random.gauss(0, noise), y + random.gauss(0, noise)))
        labels.append(lab)
    return pts, labels


def make_spiral(n=100, noise=0.08, seed=0, turns=0.8):
    """Two interleaving spiral arms, labels +1 / -1."""
    random.seed(seed)
    pts, labels = [], []
    per = n // 2
    for arm, lab in [(0, 1), (1, -1)]:
        for i in range(per):
            r = i / per
            t = turns * 2 * math.pi * r + arm * math.pi
            x = r * math.cos(t) + random.gauss(0, noise)
            y = r * math.sin(t) + random.gauss(0, noise)
            pts.append((x, y))
            labels.append(lab)
    return pts, labels


def _clf_term(s, y, loss):
    """One classification loss term for score s and label y in {-1, +1}."""
    if loss in ("hinge", "svm", "max_margin"):  # Karpathy's max-margin SVM loss
        return (1 + (-y) * s).ReLu()
    if loss == "squared_hinge":
        return (1 + (-y) * s).ReLu() ** 2
    if loss == "logistic":
        return (1 + ((-y) * s).exp()).log()
    raise ValueError(f"unknown classification loss: {loss!r} "
                     "(hinge/svm/max_margin, squared_hinge, logistic)")


def fit_classifier(pts, labels, hidden=(16, 16), epochs=120, lr=0.1, alpha=1e-4,
                   grid_res=36, frames=28, seed=1337, log_every=0, loss="hinge"):
    """Train an MLP (2 -> hidden tanh -> 1 linear) with hinge loss + L2.

    Records the decision surface on a grid at intervals for animation.
    """
    random.seed(seed)
    model = MLP(2, list(hidden) + [1])
    n = len(pts)

    # plot bounds and the grid the decision surface is evaluated on
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    x_lo, x_hi = min(xs) - 0.5, max(xs) + 0.5
    y_lo, y_hi = min(ys) - 0.5, max(ys) + 0.5
    gx = [x_lo + (x_hi - x_lo) * i / (grid_res - 1) for i in range(grid_res)]
    gy = [y_lo + (y_hi - y_lo) * i / (grid_res - 1) for i in range(grid_res)]

    def grid_scores():
        Z = np.zeros((grid_res, grid_res))
        for a, yv in enumerate(gy):
            for b, xv in enumerate(gx):
                Z[a, b] = model([Value(xv), Value(yv)]).data
        return Z

    record_every = max(1, epochs // frames)
    record = set(range(0, epochs, record_every)) | {epochs - 1}
    loss_name = loss  # keep the string; `loss` is reused below for the total Value
    losses, accs, frames = [], [], []

    for epoch in range(epochs):
        scores = [model([Value(x1), Value(x2)]) for (x1, x2) in pts]
        # every loss wants yi * score >= 1
        data_loss = sum((_clf_term(si, yi, loss_name) for si, yi in zip(scores, labels)), Value(0.0)) * (1.0 / n)
        reg = alpha * sum((p * p for p in model.parameters()), Value(0.0))
        loss = data_loss + reg

        acc = sum((s.data > 0) == (yi > 0) for s, yi in zip(scores, labels)) / n

        model.zero_grad()
        loss.backward()
        for p in model.parameters():
            p.data -= lr * p.grad

        losses.append(loss.data)
        accs.append(acc)
        if log_every and (epoch % log_every == 0 or epoch == epochs - 1):
            print(f"[micrograd] epoch {epoch:5d}  loss {loss.data:.4f}  acc {acc * 100:.0f}%")

        if epoch in record:
            frames.append({"epoch": epoch, "loss": loss.data, "acc": acc, "Z": grid_scores()})

    return {"pts": pts, "labels": labels, "gx": gx, "gy": gy,
            "losses": losses, "accs": accs, "frames": frames}


def animate_classifier(history, filename, fps=8):
    """Render a `history` from fit_classifier into an animated GIF."""
    pts, labels = history["pts"], history["labels"]
    losses, accs = history["losses"], history["accs"]
    xx, yy = np.meshgrid(np.array(history["gx"]), np.array(history["gy"]))
    px = [p[0] for p in pts]
    py = [p[1] for p in pts]
    colors = ["#d62728" if l > 0 else "#1f77b4" for l in labels]

    images = []
    for fr in history["frames"]:
        ep = fr["epoch"]
        fig, ax = plt.subplots(1, 2, figsize=(13, 6))

        # decision regions split at score = 0, with the boundary drawn on top
        ax[0].contourf(xx, yy, fr["Z"], levels=[-1e9, 0, 1e9],
                       colors=["#aec7e8", "#ff9896"], alpha=0.6)
        ax[0].contour(xx, yy, fr["Z"], levels=[0], colors="k", linewidths=1)
        ax[0].scatter(px, py, c=colors, s=18, edgecolors="k", linewidths=0.3)
        ax[0].set_xticks([]); ax[0].set_yticks([])
        ax[0].set_title(f"decision boundary (epoch {ep}, acc {fr['acc'] * 100:.0f}%)")

        ax[1].plot(range(len(losses)), losses, "m-")
        ax[1].axvline(ep, color="k", lw=0.6)
        ax[1].set_yscale("log")
        ax[1].set_xlabel("epoch")
        ax[1].set_title("loss (hinge + L2, log)")

        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=80)
        plt.close(fig)
        buf.seek(0)
        images.append(Image.open(buf).convert("RGB"))

    duration = int(1000 / fps)
    durations = [duration] * len(images)
    durations[-1] = duration * 6
    images[0].save(filename, save_all=True, append_images=images[1:],
                   duration=durations, loop=0)
    return filename
