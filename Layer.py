from operator import neg
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

    def backward(self, input, dvalues):
        self.dweights = np.dot(input.T, dvalues)
        self.dbiases = np.sum(dvalues, axis = 0, keepdims = True)
        
        dinput = np.dot(dvalues, self.weights.T)

        return dinput

    def activate(self, x):
        # Apply the activation function to the layer output
        if self.activation == 'relu':
            return self.__ReLu(x)
        elif self.activation == 'softmax':
            return self.__Softmax(x)
            
    # Define private method for the activation functions
    def __ReLu(self, x):
        return np.maximum(0, x)

    def __diff_ReLu(self, x):
        return 1 if x > 0 else 0

    def __Softmax(self, x):
        exp_vals = np.exp(x - np.max(x, axis = 1, keepdims = True))
        probs = exp_vals / np.sum(exp_vals, axis = 1, keepdims = True)
        return probs

    def CategoricalCrossEntropy(self, y_true, y_hat):
        y_hat_clipped = np.clip(y_hat, 1e-7, 1-1e-7)
        samples = len(y_true)

        if len(y_true.shape) == 1:
            correct_confidences = y_hat_clipped[range(samples), y_true]
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_hat_clipped*y_true, axis = 1)

        neg_log_likelihoods = -np.log(correct_confidences)
        return np.mean(neg_log_likelihoods)
