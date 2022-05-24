from Neuron import *
from Layer import *
from nnfs.datasets import spiral_data
from NNModel import *

if __name__ == '__main__':
    X, y = spiral_data(100, 3)

    dense1 = Layer(2, 3, activation = 'relu')
    dense2 = Layer(3, 3, activation = 'softmax')

    dense1.forward(X)
    dense2.forward(dense1.output)

    loss_ly = dense2.CategoricalCrossEntropy(y, dense2.output)

    model = NNModel()
    model.add(dense1)
    model.add(dense2)
    model.compile()

    acc, loss = model.fit(X,y)


    print(loss_ly, loss, acc)

    