from Neuron import *
from Layer import *
if __name__ == '__main__':

    X = np.array([
        [1, 2, 3, 2.5],
        [2, 5, -1, 2],
        [-1.5, 2.7, 3.3, -0.8]
    ])

    layer1 = Layer(4, 5)
    layer2 = Layer(5, 2)

    layer1.forward(X)
    layer2.forward(layer1.output)
    print(layer2.output)