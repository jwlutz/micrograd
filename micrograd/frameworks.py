"""Run the same training tasks on PyTorch and JAX, for comparison with micrograd.

These mirror the micrograd setup: full-batch gradient descent, tanh hidden
layers, linear output, configurable loss (mse/mae/huber for regression,
hinge/squared_hinge/logistic for classification), and uniform(-1, 1) init to
match. They print progress and report the final metric and wall-clock time.
Used by the CLI's --framework flag and by benchmark.py.
"""

import time


# ---------------------------------------------------------------- PyTorch ----

def torch_regression(xs, ys, hidden, epochs, lr, seed=0, log_every=0, loss="mse"):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    torch.manual_seed(seed)
    X = torch.tensor(xs, dtype=torch.float32).view(-1, 1)
    Y = torch.tensor(ys, dtype=torch.float32).view(-1, 1)

    layers, prev = [], 1
    for h in hidden:
        layers += [nn.Linear(prev, h), nn.Tanh()]
        prev = h
    layers += [nn.Linear(prev, 1)]
    model = nn.Sequential(*layers)
    for layer in model:  # match micrograd/jax uniform(-1, 1) init for a fair comparison
        if isinstance(layer, nn.Linear):
            nn.init.uniform_(layer.weight, -1.0, 1.0)
            nn.init.uniform_(layer.bias, -1.0, 1.0)
    opt = torch.optim.SGD(model.parameters(), lr=lr)

    def crit(pred):
        if loss == "mse":
            return ((pred - Y) ** 2).mean()
        if loss == "mae":
            return (pred - Y).abs().mean()
        if loss == "huber":
            return F.huber_loss(pred, Y, delta=1.0)
        raise ValueError(f"unknown regression loss: {loss!r}")

    t0, l = time.time(), None
    for ep in range(epochs):
        l = crit(model(X))
        opt.zero_grad(); l.backward(); opt.step()
        if log_every and ep % log_every == 0:
            print(f"[pytorch] epoch {ep:5d}  loss {l.item():.5f}")
    print(f"[pytorch] final loss {l.item():.5f}  ({time.time() - t0:.2f}s)")
    return l.item()


def torch_classification(pts, labels, hidden, epochs, lr, alpha=1e-4, seed=0, log_every=0, loss="hinge"):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    torch.manual_seed(seed)
    X = torch.tensor(pts, dtype=torch.float32)
    Y = torch.tensor(labels, dtype=torch.float32).view(-1, 1)

    layers, prev = [], 2
    for h in hidden:
        layers += [nn.Linear(prev, h), nn.Tanh()]
        prev = h
    layers += [nn.Linear(prev, 1)]
    model = nn.Sequential(*layers)
    for layer in model:  # match micrograd/jax uniform(-1, 1) init for a fair comparison
        if isinstance(layer, nn.Linear):
            nn.init.uniform_(layer.weight, -1.0, 1.0)
            nn.init.uniform_(layer.bias, -1.0, 1.0)
    opt = torch.optim.SGD(model.parameters(), lr=lr)

    def data_loss(score):
        if loss in ("hinge", "svm", "max_margin"):
            return torch.clamp(1 - Y * score, min=0).mean()
        if loss == "squared_hinge":
            return torch.clamp(1 - Y * score, min=0).pow(2).mean()
        if loss == "logistic":
            return F.softplus(-Y * score).mean()
        raise ValueError(f"unknown classification loss: {loss!r}")

    t0, acc = time.time(), 0.0
    for ep in range(epochs):
        score = model(X)
        l = data_loss(score) + alpha * sum((p ** 2).sum() for p in model.parameters())
        acc = ((score > 0) == (Y > 0)).float().mean().item()
        opt.zero_grad(); l.backward(); opt.step()
        if log_every and ep % log_every == 0:
            print(f"[pytorch] epoch {ep:5d}  loss {l.item():.4f}  acc {acc * 100:.0f}%")
    print(f"[pytorch] final acc {acc * 100:.0f}%  ({time.time() - t0:.2f}s)")
    return acc


# -------------------------------------------------------------------- JAX ----

def _jax_init(hidden, nin, nout, seed):
    import jax
    sizes = [nin] + list(hidden) + [nout]
    key = jax.random.PRNGKey(seed)
    params = []
    for a, b in zip(sizes[:-1], sizes[1:]):
        key, kw, kb = jax.random.split(key, 3)
        W = jax.random.uniform(kw, (b, a), minval=-1.0, maxval=1.0)
        bias = jax.random.uniform(kb, (b,), minval=-1.0, maxval=1.0)
        params.append((W, bias))
    nonlins = [True] * (len(sizes) - 2) + [False]
    return params, nonlins


def _jax_forward(params, nonlins, X):
    import jax.numpy as jnp
    a = X
    for (W, b), nl in zip(params, nonlins):
        z = a @ W.T + b
        a = jnp.tanh(z) if nl else z
    return a


def jax_regression(xs, ys, hidden, epochs, lr, seed=0, log_every=0, loss="mse"):
    import jax
    import jax.numpy as jnp
    params, nonlins = _jax_init(hidden, 1, 1, seed)
    X = jnp.array(xs).reshape(-1, 1)
    Y = jnp.array(ys).reshape(-1, 1)

    def crit(p):
        pred = _jax_forward(p, nonlins, X)
        if loss == "mse":
            return jnp.mean((pred - Y) ** 2)
        if loss == "mae":
            return jnp.mean(jnp.abs(pred - Y))
        if loss == "huber":
            r = pred - Y
            a = jnp.abs(r)
            return jnp.mean(jnp.where(a <= 1.0, 0.5 * r ** 2, a - 0.5))
        raise ValueError(f"unknown regression loss: {loss!r}")

    step = jax.jit(jax.value_and_grad(crit))

    t0, l = time.time(), None
    for ep in range(epochs):
        l, grads = step(params)
        params = [(W - lr * gW, b - lr * gb) for (W, b), (gW, gb) in zip(params, grads)]
        if log_every and ep % log_every == 0:
            print(f"[jax] epoch {ep:5d}  loss {float(l):.5f}")
    print(f"[jax] final loss {float(l):.5f}  ({time.time() - t0:.2f}s)")
    return float(l)


def jax_classification(pts, labels, hidden, epochs, lr, alpha=1e-4, seed=0, log_every=0, loss="hinge"):
    import jax
    import jax.numpy as jnp
    params, nonlins = _jax_init(hidden, 2, 1, seed)
    X = jnp.array(pts)
    Y = jnp.array(labels).reshape(-1, 1)

    def loss_fn(p):
        score = _jax_forward(p, nonlins, X)
        if loss in ("hinge", "svm", "max_margin"):
            data = jnp.mean(jnp.maximum(0.0, 1 - Y * score))
        elif loss == "squared_hinge":
            data = jnp.mean(jnp.maximum(0.0, 1 - Y * score) ** 2)
        elif loss == "logistic":
            data = jnp.mean(jnp.logaddexp(0.0, -Y * score))
        else:
            raise ValueError(f"unknown classification loss: {loss!r}")
        reg = alpha * sum(jnp.sum(W ** 2) + jnp.sum(b ** 2) for (W, b) in p)
        return data + reg

    def accuracy(p):
        return float(jnp.mean((_jax_forward(p, nonlins, X) > 0) == (Y > 0)))

    step = jax.jit(jax.value_and_grad(loss_fn))

    t0, acc = time.time(), 0.0
    for ep in range(epochs):
        l, grads = step(params)
        params = [(W - lr * gW, b - lr * gb) for (W, b), (gW, gb) in zip(params, grads)]
        if log_every and ep % log_every == 0:
            acc = accuracy(params)
            print(f"[jax] epoch {ep:5d}  loss {float(l):.4f}  acc {acc * 100:.0f}%")
    acc = accuracy(params)
    print(f"[jax] final acc {acc * 100:.0f}%  ({time.time() - t0:.2f}s)")
    return acc
