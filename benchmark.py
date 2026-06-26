"""Compare micrograd vs PyTorch vs JAX on the same tasks.

Trains each engine with identical architecture and hyperparameters, records the
final metric and wall-clock time, and writes a Markdown table to BENCHMARKS.md.
The metrics agree (same math); the times show micrograd is the slow, transparent
teaching engine and the frameworks are the fast production ones.

Run from the project root:  python benchmark.py
"""

import math
import time

from micrograd import classify, frameworks, learn


def _timed(fn, *args, **kwargs):
    t0 = time.time()
    out = fn(*args, **kwargs)
    return out, time.time() - t0


def bench_regression(name, target, domain, epochs, lr, hidden):
    xs = [domain[0] + (domain[1] - domain[0]) * i / 39 for i in range(40)]
    ys = [target(x) for x in xs]
    rows = []
    h, t = _timed(learn.fit_function, target=target, domain=domain,
                  hidden=hidden[0], epochs=epochs, lr=lr)
    rows.append(("micrograd", h["losses"][-1], t))
    m, t = _timed(frameworks.torch_regression, xs, ys, hidden, epochs, lr)
    rows.append(("pytorch", m, t))
    m, t = _timed(frameworks.jax_regression, xs, ys, hidden, epochs, lr)
    rows.append(("jax", m, t))
    return (name, "MSE", epochs, hidden, rows)


def bench_classification(name, maker, epochs, lr, hidden):
    pts, labels = maker(n=120, noise=0.1, seed=1337)
    rows = []
    h, t = _timed(classify.fit_classifier, pts, labels,
                  hidden=hidden, epochs=epochs, lr=lr, frames=1)  # frames=1: skip animation grid
    rows.append(("micrograd", h["accs"][-1], t))
    m, t = _timed(frameworks.torch_classification, pts, labels, hidden, epochs, lr)
    rows.append(("pytorch", m, t))
    m, t = _timed(frameworks.jax_classification, pts, labels, hidden, epochs, lr)
    rows.append(("jax", m, t))
    return (name, "accuracy", epochs, hidden, rows)


def _fmt(metric_name, v):
    return f"{v * 100:.0f}%" if metric_name == "accuracy" else f"{v:.4f}"


def main():
    results = [
        bench_regression("sin", math.sin, (-math.pi, math.pi), 1000, 0.08, (12,)),
        bench_classification("moons", classify.make_moons, 120, 0.10, (16, 16)),
    ]

    lines = [
        "# Benchmarks: micrograd vs PyTorch vs JAX",
        "",
        "Same architecture and hyperparameters on every engine. The metrics agree",
        "(it's the same math); the wall-clock times show micrograd is the slow,",
        "transparent teaching engine while PyTorch and JAX are built for speed.",
        "Gradient-level agreement is asserted in `tests/test_frameworks.py`.",
        "",
    ]
    for name, metric_name, epochs, hidden, rows in results:
        mg_time = next(t for eng, _, t in rows if eng == "micrograd")
        lines += [
            f"## {name}  ({metric_name}, {epochs} epochs, hidden={list(hidden)})",
            "",
            f"| engine | {metric_name} | time | speedup vs micrograd |",
            "| --- | --- | --- | --- |",
        ]
        for eng, v, t in rows:
            speed = "1x" if eng == "micrograd" else f"{mg_time / t:.0f}x"
            lines.append(f"| {eng} | {_fmt(metric_name, v)} | {t:.2f}s | {speed} |")
        lines.append("")

    text = "\n".join(lines)
    with open("BENCHMARKS.md", "w") as f:
        f.write(text)
    print(text)


if __name__ == "__main__":
    main()
