from micrograd.engine import Value
import random

class Module:

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0

    def parameters(self):
        return []

class Neuron(Module):

    def __init__(self, nin, nonlin=True): #nin is number of inputs
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1, 1))
        self.nonlin = nonlin #apply tanh? False = linear output (for regression)

    def __call__(self, x):
        # w * x + b
        act = sum((wi*xi for wi, xi in zip(self.w, x)), self.b) #activation function
        return act.tanh() if self.nonlin else act #squish, unless this is a linear layer
    
    def parameters(self):
        return self.w + [self.b]
    
class Layer(Module):

    def __init__(self, nin, nout, nonlin=True):
        self.neurons = [Neuron(nin, nonlin=nonlin) for _ in range(nout)]

    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs
    
    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]
    
class MLP(Module):
    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        # hidden layers use tanh; the final layer is linear (good for regression)
        self.layers = [Layer(sz[i], sz[i+1], nonlin=(i != len(nouts) - 1)) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]