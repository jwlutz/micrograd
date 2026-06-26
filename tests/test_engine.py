import torch
from micrograd.engine import Value


def test_sanity_check():
    # one input, a small mix of ops, compared against PyTorch
    x = Value(-4.0)
    z = 2 * x + 2 + x
    q = z.ReLu() + z * x
    h = (z * z).ReLu()
    y = h + q + q * x
    y.backward()
    xmg, ymg = x, y

    x = torch.Tensor([-4.0]).double()
    x.requires_grad = True
    z = 2 * x + 2 + x
    q = z.relu() + z * x
    h = (z * z).relu()
    y = h + q + q * x
    y.backward()
    xpt, ypt = x, y

    # forward pass should match
    assert abs(ymg.data - ypt.data.item()) < 1e-6
    # backward pass should match
    assert abs(xmg.grad - xpt.grad.item()) < 1e-6


def test_more_ops():
    # two inputs, every operator including the reverse ops you just wrote
    a = Value(-4.0)
    b = Value(2.0)
    c = a + b
    d = a * b + b**3
    c = c + c + 1
    c = c + 1 + c + (-a)
    d = d + d * 2 + (b + a).ReLu()
    d = d + 3 * d + (b - a).ReLu()
    e = c - d
    f = e**2
    g = f / 2.0
    g = g + 10.0 / f        # exercises __rtruediv__
    g = g + (2.0 - a)       # exercises __rsub__
    g.backward()
    amg, bmg, gmg = a, b, g

    a = torch.Tensor([-4.0]).double(); a.requires_grad = True
    b = torch.Tensor([2.0]).double(); b.requires_grad = True
    c = a + b
    d = a * b + b**3
    c = c + c + 1
    c = c + 1 + c + (-a)
    d = d + d * 2 + (b + a).relu()
    d = d + 3 * d + (b - a).relu()
    e = c - d
    f = e**2
    g = f / 2.0
    g = g + 10.0 / f
    g = g + (2.0 - a)
    g.backward()
    apt, bpt, gpt = a, b, g

    tol = 1e-6
    # forward pass should match
    assert abs(gmg.data - gpt.data.item()) < tol
    # backward pass should match
    assert abs(amg.grad - apt.grad.item()) < tol
    assert abs(bmg.grad - bpt.grad.item()) < tol


def test_tanh_exp():
    # the two activations Karpathy's tests don't cover
    x = Value(0.8)
    y = Value(-1.5)
    z = (x * y).tanh() + (x + y).exp() - x.exp()
    z.backward()
    xmg, ymg, zmg = x, y, z

    x = torch.Tensor([0.8]).double(); x.requires_grad = True
    y = torch.Tensor([-1.5]).double(); y.requires_grad = True
    z = (x * y).tanh() + (x + y).exp() - x.exp()
    z.backward()
    xpt, ypt, zpt = x, y, z

    tol = 1e-6
    assert abs(zmg.data - zpt.data.item()) < tol
    assert abs(xmg.grad - xpt.grad.item()) < tol
    assert abs(ymg.grad - ypt.grad.item()) < tol


def test_log_abs():
    # log and abs, used by the logistic and MAE losses
    a = Value(1.7)
    b = Value(-2.3)
    z = a.log() + abs(b) * a - abs(a)
    z.backward()
    amg, bmg, zmg = a, b, z

    a = torch.Tensor([1.7]).double(); a.requires_grad = True
    b = torch.Tensor([-2.3]).double(); b.requires_grad = True
    z = a.log() + b.abs() * a - a.abs()
    z.backward()
    apt, bpt, zpt = a, b, z

    tol = 1e-6
    assert abs(zmg.data - zpt.data.item()) < tol
    assert abs(amg.grad - apt.grad.item()) < tol
    assert abs(bmg.grad - bpt.grad.item()) < tol
