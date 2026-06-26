"""Cross-framework gradient check at the MLP level.

`test_engine.py` verifies individual ops against PyTorch. This goes one level
up: it builds a *fixed* network (fixed weights, fixed input), then confirms the
forward output and every parameter's gradient match PyTorch and JAX. Using
fixed weights (not a trained net) keeps the comparison deterministic, so the
numbers must agree exactly. This verifies that nn.py composes correctly.
"""

import random

import pytest

from micrograd.engine import Value
from micrograd.nn import MLP

X = [0.5, -0.3]   # fixed input
Y = 1.0           # fixed target
ARCH = [4, 4, 1]  # 2 -> 4 (tanh) -> 4 (tanh) -> 1 (linear)


def build_micrograd():
    """Forward + backward a fixed micrograd MLP; return its output and the
    weights/gradients structured per layer."""
    random.seed(42)
    model = MLP(2, ARCH)

    out = model([Value(X[0]), Value(X[1])])
    loss = (out - Y) ** 2
    model.zero_grad()
    loss.backward()

    layers = []
    for layer in model.layers:
        layers.append({
            "W": [[w.data for w in neuron.w] for neuron in layer.neurons],
            "b": [neuron.b.data for neuron in layer.neurons],
            "gW": [[w.grad for w in neuron.w] for neuron in layer.neurons],
            "gb": [neuron.b.grad for neuron in layer.neurons],
            "nonlin": layer.neurons[0].nonlin,
        })
    return out.data, layers


def test_mlp_matches_pytorch():
    torch = pytest.importorskip("torch")
    out_mg, layers = build_micrograd()

    x = torch.tensor(X, dtype=torch.double)
    Ws = [torch.tensor(L["W"], dtype=torch.double, requires_grad=True) for L in layers]
    bs = [torch.tensor(L["b"], dtype=torch.double, requires_grad=True) for L in layers]

    a = x
    for i, L in enumerate(layers):
        z = Ws[i] @ a + bs[i]
        a = torch.tanh(z) if L["nonlin"] else z
    out = a[0]
    loss = (out - Y) ** 2
    loss.backward()

    assert abs(out.item() - out_mg) < 1e-9
    for i, L in enumerate(layers):
        for r in range(len(L["gW"])):
            for c in range(len(L["gW"][r])):
                assert abs(Ws[i].grad[r][c].item() - L["gW"][r][c]) < 1e-6
            assert abs(bs[i].grad[r].item() - L["gb"][r]) < 1e-6


def test_mlp_matches_jax():
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)  # float64 to match within 1e-6
    import jax.numpy as jnp

    out_mg, layers = build_micrograd()

    params = [(jnp.array(L["W"]), jnp.array(L["b"])) for L in layers]
    nonlins = [L["nonlin"] for L in layers]
    x = jnp.array(X)

    def forward(params):
        a = x
        for (W, b), nonlin in zip(params, nonlins):
            z = W @ a + b
            a = jnp.tanh(z) if nonlin else z
        return a[0]

    out_val = float(forward(params))
    grads = jax.grad(lambda p: (forward(p) - Y) ** 2)(params)

    assert abs(out_val - out_mg) < 1e-6
    for i, L in enumerate(layers):
        gW, gb = grads[i]
        for r in range(len(L["gW"])):
            for c in range(len(L["gW"][r])):
                assert abs(float(gW[r][c]) - L["gW"][r][c]) < 1e-6
            assert abs(float(gb[r]) - L["gb"][r]) < 1e-6
