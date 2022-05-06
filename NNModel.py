from Layer import *
class NNModel:
    def __init__(self):
        self.layers = []

    def add(self, layer: Layer):
        self.layers.append(layer)