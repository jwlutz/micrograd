# Benchmarks: micrograd vs PyTorch vs JAX

Same architecture and hyperparameters on every engine. The metrics agree
(it's the same math); the wall-clock times show micrograd is the slow,
transparent teaching engine while PyTorch and JAX are built for speed.
Gradient-level agreement is asserted in `tests/test_frameworks.py`.

## sin  (MSE, 1000 epochs, hidden=[12])

| engine | MSE | time | speedup vs micrograd |
| --- | --- | --- | --- |
| micrograd | 0.0032 | 5.32s | 1x |
| pytorch | 0.0028 | 0.92s | 6x |
| jax | 0.0016 | 0.66s | 8x |

## moons  (accuracy, 120 epochs, hidden=[16, 16])

| engine | accuracy | time | speedup vs micrograd |
| --- | --- | --- | --- |
| micrograd | 99% | 57.77s | 1x |
| pytorch | 100% | 0.02s | 2520x |
| jax | 99% | 0.59s | 97x |
