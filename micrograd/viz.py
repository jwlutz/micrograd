"""Graphviz visualization of the autograd computation graph."""

import io

from graphviz import Digraph
from PIL import Image


def trace(root):
    """Walk the graph backwards from `root`, collecting every node and edge."""
    nodes, edges = set(), set()

    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v._prev:
                edges.add((child, v))
                build(child)

    build(root)
    return nodes, edges


def _grad_color(grad, max_grad):
    """Pick a fill color from a node's gradient.

    White for zero, green for positive, red for negative. The magnitude is
    scaled relative to `max_grad`, so the most influential node is fully
    saturated and the rest fade toward white.
    """
    if max_grad == 0:
        return "#ffffff"
    intensity = abs(grad) / max_grad      # 0.0 (none) .. 1.0 (biggest in graph)
    fade = int(255 * (1 - intensity))     # 255 (white) .. 0 (full color)
    if grad > 0:
        return "#%02x%02x%02x" % (fade, 255, fade)   # white -> green
    if grad < 0:
        return "#%02x%02x%02x" % (255, fade, fade)   # white -> red
    return "#ffffff"


def draw_dot(root, format="svg", rankdir="LR", max_grad=None):
    """Return a graphviz Digraph of the computation graph ending at `root`.

    Nodes are filled by gradient: green = positive, red = negative, deeper
    color = larger magnitude. Pass `max_grad` to fix the color scale (used by
    the animation so colors don't rescale every frame); otherwise it is taken
    from the largest gradient currently in the graph. rankdir "LR" draws
    left-to-right, "TB" top-to-bottom.
    """
    dot = Digraph(format=format, graph_attr={"rankdir": rankdir})

    nodes, edges = trace(root)
    if max_grad is None:
        max_grad = max((abs(n.grad) for n in nodes), default=0.0)

    for n in nodes:
        uid = str(id(n))
        # one rectangular node per Value, filled by how big its gradient is
        dot.node(
            name=uid,
            label="{ %s | data %.4f | grad %.4f }" % (n.label, n.data, n.grad),
            shape="record",
            style="filled",
            fillcolor=_grad_color(n.grad, max_grad),
        )
        if n._op:
            # if this Value is the result of an op, draw a small op node feeding it
            dot.node(name=uid + n._op, label=n._op)
            dot.edge(uid + n._op, uid)

    for n1, n2 in edges:
        # connect each input to the op node of its result
        dot.edge(str(id(n1)), str(id(n2)) + n2._op)

    return dot


def _topo(root):
    """Topological order of the graph: every node comes after its children.
    Same ordering the engine's backward() uses."""
    order, visited = [], set()

    def build(v):
        if v not in visited:
            visited.add(v)
            for child in v._prev:
                build(child)
            order.append(v)

    build(root)
    return order


def animate_backward(root, filename, rankdir="LR", duration=700):
    """Render the backward pass step by step into an animated GIF.

    One frame per node, played as gradients propagate from the output back to
    the inputs. Colors are locked to the final gradient scale, so each node
    grows toward its end color instead of the scale jumping around.
    """
    order = _topo(root)

    # 1. run the full backward once just to learn the final color scale
    for n in order:
        n.grad = 0.0
    root.grad = 1.0
    for node in reversed(order):
        node._backward()
    final_max = max((abs(n.grad) for n in order), default=0.0)

    # 2. reset, then replay one step at a time, snapshotting each state
    for n in order:
        n.grad = 0.0

    frames = []

    def snapshot():
        png = draw_dot(root, format="png", rankdir=rankdir, max_grad=final_max).pipe(format="png")
        frames.append(Image.open(io.BytesIO(png)).convert("RGB"))

    snapshot()                       # all grads 0: forward pass done, nothing flowed yet
    root.grad = 1.0
    snapshot()                       # seed the output's gradient = 1
    for node in reversed(order):
        node._backward()
        snapshot()                   # gradients spread one node at a time

    # 3. pad every frame to the same size (layout shifts a hair as numbers change)
    w = max(f.width for f in frames)
    h = max(f.height for f in frames)
    padded = []
    for f in frames:
        canvas = Image.new("RGB", (w, h), "white")
        canvas.paste(f, (0, 0))
        padded.append(canvas)

    # 4. write the GIF, holding the final frame longer
    durations = [duration] * len(padded)
    durations[-1] = duration * 4
    padded[0].save(
        filename,
        save_all=True,
        append_images=padded[1:],
        duration=durations,
        loop=0,
    )
    return filename
