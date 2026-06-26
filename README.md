# micrograd

A from-scratch scalar autograd engine and a small neural-net library on top of it, with tools to draw the gradients and watch a network train. My version of [Karpathy's micrograd](https://github.com/karpathy/micrograd).

![a network learning sin(x)](assets/learn_sin.gif)

## What's here

| file | what it does |
| --- | --- |
| `micrograd/engine.py` | the `Value` autograd engine |
| `micrograd/nn.py` | `Neuron`, `Layer`, `MLP` |
| `micrograd/viz.py` | draw the graph, animate the backward pass |
| `micrograd/learn.py` | fit a 1-D function, with a training dashboard |
| `micrograd/classify.py` | 2-D classifier with an animated boundary |
| `micrograd/frameworks.py` | the same training in PyTorch and JAX |
| `micrograd/cli.py` | command-line runner |

## The engine

Ops: `+ - * / **`, `tanh`, `exp`, `log`, `relu`, `abs`, each with its own backward.

```python
from micrograd import Value

a = Value(-4.0)
b = Value(2.0)
c = a * b + b**3      # build an expression
c.backward()          # backprop
print(a.grad, b.grad) # 2.0  8.0
```

Gradients match PyTorch and JAX to `1e-6` (`tests/`).

## Backprop, step by step

`viz.py` colors each node by its gradient and replays the backward pass one node at a time.

![graph](assets/graph.png)

![backprop](assets/backward.gif)

## A network learning a function

`learn.py` trains an MLP and records the fit, the per-neuron contributions, the residual, the loss, and the gradient/weight sizes.

| sin | abs | step |
| --- | --- | --- |
| ![sin](assets/learn_sin.gif) | ![abs](assets/learn_abs.gif) | ![step](assets/learn_step.gif) |

## Classifiers

| moons | spiral |
| --- | --- |
| ![moons](assets/moons.gif) | ![spiral](assets/spiral.gif) |

## CLI

```bash
python -m micrograd.cli --list
python -m micrograd.cli --task sin --gif out.gif
python -m micrograd.cli --task moons --loss svm
python -m micrograd.cli --task spiral --framework pytorch
```

Flags: `--task`, `--framework` (micrograd/pytorch/jax), `--loss`, `--lr`, `--epochs`, `--hidden`, `--alpha`, `--seed`.

## Tests and benchmarks

```bash
pytest               # gradients vs PyTorch and JAX
python benchmark.py  # writes BENCHMARKS.md
```

## Setup

```bash
pip install -e .
pip install -r requirements.txt   # matplotlib, pillow, graphviz, torch, jax, pytest
brew install graphviz             # system binary for graph drawing
```

## Credit

Based on [Karpathy's micrograd](https://github.com/karpathy/micrograd) and his [neural networks: zero to hero](https://karpathy.ai/zero-to-hero.html) series.
