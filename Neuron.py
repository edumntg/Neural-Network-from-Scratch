import numpy as np
class Neuron:
    def __init__(self, n: int):
        self.size = n

        # Initialize weights
        self.weights = np.random.randn(n)
        self.bias = np.random.randn(n,1)

