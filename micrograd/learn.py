"""Watch a micrograd MLP learn a 1-D function.

`fit_function` trains a single-hidden-layer MLP on a target like sin(x) and
records everything needed to visualize training. `animate_fit` turns that into
an animated dashboard GIF: the learning view plus four diagnostic panels.
"""

import io
import math
import random

import matplotlib
matplotlib.use("Agg")  # render to memory buffers, no display window
import matplotlib.pyplot as plt
from PIL import Image

from micrograd.engine import Value
from micrograd.nn import MLP


def _reg_term(r, loss, delta=1.0):
    """One regression loss term for a residual r = prediction - target."""
    if loss == "mse":
        return r * r
    if loss == "mae":
        return abs(r)
    if loss == "huber":
        if abs(r.data) <= delta:
            return 0.5 * r * r
        return delta * (abs(r) - 0.5 * delta)
    raise ValueError(f"unknown regression loss: {loss!r} (mse, mae, huber)")


def fit_function(target=math.sin, domain=(-math.pi, math.pi), n_points=40,
                 hidden=12, epochs=2000, lr=0.08, record_every=50, seed=1337, log_every=0,
                 loss="mse"):
    """Train a 1-hidden-layer MLP (1 -> hidden tanh -> 1 linear) to fit `target`.

    Returns a `history` dict holding the data, per-epoch diagnostics, and a list
    of frame snapshots for the animation.
    """
    random.seed(seed)
    lo, hi = domain

    # training points and a finer grid for smooth plotting
    xs = [lo + (hi - lo) * i / (n_points - 1) for i in range(n_points)]
    ys = [target(x) for x in xs]
    grid = [lo + (hi - lo) * i / 199 for i in range(200)]
    grid_target = [target(x) for x in grid]

    model = MLP(1, [hidden, 1])

    def predict_grid():
        return [model([Value(x)]).data for x in grid]

    def decompose_grid():
        # each hidden neuron's contribution to the output: w_out_j * tanh(neuron_j(x))
        hidden_layer, out_layer = model.layers[0], model.layers[1]
        out_neuron = out_layer.neurons[0]
        rows = []
        for x in grid:
            acts = hidden_layer([Value(x)])  # list of `hidden` tanh activations
            rows.append([out_neuron.w[j].data * acts[j].data for j in range(len(acts))])
        return rows

    loss_name = loss  # keep the string; the loop reuses `loss` for the total Value
    losses, grad_mag, weight_mag = [], [], []
    frames = []

    # record more frames early (learning is fast) and fewer later (slow refinement)
    record_epochs = set(range(0, min(150, epochs), 6)) | set(range(0, epochs, record_every))
    record_epochs.add(epochs - 1)

    for epoch in range(epochs):
        # forward over all points
        preds = [model([Value(x)]) for x in xs]
        loss = sum((_reg_term(p - y, loss_name) for p, y in zip(preds, ys)), Value(0.0)) * (1.0 / n_points)

        model.zero_grad()
        loss.backward()

        params = model.parameters()
        losses.append(loss.data)
        grad_mag.append(sum(abs(p.grad) for p in params) / len(params))
        weight_mag.append(sum(abs(p.data) for p in params) / len(params))

        # SGD step
        for p in params:
            p.data -= lr * p.grad

        if log_every and (epoch % log_every == 0 or epoch == epochs - 1):
            print(f"[micrograd] epoch {epoch:5d}  loss {loss.data:.5f}")

        if epoch in record_epochs:
            pred = predict_grid()
            frames.append({
                "epoch": epoch,
                "loss": losses[-1],
                "pred": pred,
                "contribs": decompose_grid(),
                "residual": [pg - tg for pg, tg in zip(pred, grid_target)],
            })

    return {
        "xs": xs, "ys": ys, "grid": grid, "grid_target": grid_target,
        "losses": losses, "grad_mag": grad_mag, "weight_mag": weight_mag,
        "hidden": hidden, "lr": lr, "frames": frames,
    }


def animate_fit(history, filename, fps=8):
    """Render a `history` from fit_function into an animated dashboard GIF."""
    grid, xs, ys = history["grid"], history["xs"], history["ys"]
    grid_target = history["grid_target"]
    H = history["hidden"]

    # adaptive y-limits so non-sin targets aren't clipped
    tmin, tmax = min(grid_target), max(grid_target)
    pad = 0.3 * (tmax - tmin) + 0.2
    ylo, yhi = tmin - pad, tmax + pad
    rlim = (yhi - ylo) / 2
    losses, grad_mag, weight_mag = history["losses"], history["grad_mag"], history["weight_mag"]

    images = []
    for fr in history["frames"]:
        ep = fr["epoch"]
        fig, ax = plt.subplots(2, 3, figsize=(15, 8))

        # 1. learning view: target vs current network output
        ax[0, 0].plot(grid, grid_target, "k--", lw=2, label="target")
        ax[0, 0].plot(grid, fr["pred"], "b-", lw=2, label="network")
        ax[0, 0].scatter(xs, ys, c="k", s=12, alpha=0.35)
        ax[0, 0].set_ylim(ylo, yhi)
        ax[0, 0].set_title(f"learning the function (epoch {ep})")
        ax[0, 0].legend(loc="upper right")

        # 2. per-neuron decomposition: each hidden unit's contribution
        for j in range(H):
            ax[0, 1].plot(grid, [fr["contribs"][i][j] for i in range(len(grid))], lw=1)
        ax[0, 1].set_ylim(ylo, yhi)
        ax[0, 1].set_title(f"per-neuron contributions ({H} units)")

        # 3. residual: where the network is still wrong
        ax[0, 2].plot(grid, fr["residual"], "r-")
        ax[0, 2].axhline(0, color="k", lw=0.5)
        ax[0, 2].set_ylim(-rlim, rlim)
        ax[0, 2].set_title("residual (network - target)")

        # 4. loss curve (log scale), with a marker at the current epoch
        ax[1, 0].plot(range(len(losses)), losses, "m-")
        ax[1, 0].axvline(ep, color="k", lw=0.6)
        ax[1, 0].set_yscale("log")
        ax[1, 0].set_xlabel("epoch")
        ax[1, 0].set_title("loss (MSE, log scale)")

        # 5. gradient and weight magnitudes over training
        ax[1, 1].plot(range(len(grad_mag)), grad_mag, label="mean |grad|")
        ax[1, 1].plot(range(len(weight_mag)), weight_mag, label="mean |weight|")
        ax[1, 1].axvline(ep, color="k", lw=0.6)
        ax[1, 1].set_yscale("log")
        ax[1, 1].set_xlabel("epoch")
        ax[1, 1].set_title("gradient / weight magnitude (log)")
        ax[1, 1].legend(loc="upper right")

        # 6. text panel
        ax[1, 2].axis("off")
        ax[1, 2].text(
            0.0, 0.5,
            f"epoch:  {ep}\n"
            f"loss:   {fr['loss']:.5f}\n"
            f"arch:   1 -> {H} (tanh) -> 1 (linear)\n"
            f"lr:     {history['lr']}",
            fontsize=13, family="monospace", va="center",
        )

        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=80)
        plt.close(fig)
        buf.seek(0)
        images.append(Image.open(buf).convert("RGB"))

    duration = int(1000 / fps)
    durations = [duration] * len(images)
    durations[-1] = duration * 6  # hold the final frame
    images[0].save(filename, save_all=True, append_images=images[1:],
                   duration=durations, loop=0)
    return filename
