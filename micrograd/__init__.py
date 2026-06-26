"""micrograd: a tiny scalar autograd engine and neural-net library.

The core (engine, nn) is pure Python. The demo modules (viz, learn, classify,
frameworks) pull in optional deps (graphviz, matplotlib, pillow, torch, jax) and
are imported explicitly, so `import micrograd` stays lightweight.
"""

from micrograd.engine import Value
from micrograd.nn import Layer, MLP, Module, Neuron

__all__ = ["Value", "Module", "Neuron", "Layer", "MLP"]
