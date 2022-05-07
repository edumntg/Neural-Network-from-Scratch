from Neuron import *

np.random.seed(0)

class Layer:
    def __init__(self, input_size: int, n_neurons: int, activation = 'relu'):

        # Initialize neurons
        self.layer_size = n_neurons
        self.input_size = input_size
        self.activation = activation.lower()

        # Initi weights
        self.weights = 0.10*np.random.randn(input_size, n_neurons)
        self.bias = np.zeros((1, n_neurons))


    def forward(self, input):
        #assert self.weights.shape[1] == input.shape[0]

        output = np.dot(input, self.weights) + self.bias
        output = self.activate(output)
        self.output = output

        return output

    def activate(self, x):
        # Apply the activation function to the layer output
        if self.activation == 'relu':
            return self.__ReLu(x)
        elif self.activation == 'softmax':
            return self.__Softmax(x)
            
    # Define private method for the activation functions
    def __ReLu(self, x):
        return np.maximum(0, x)

    def __Softmax(self, x):
        exp_vals = np.exp(x - np.max(x, axis = 1, keepdims = True))
        probs = exp_vals / np.sum(exp_vals, axis = 1, keepdims = True)
        return probs
