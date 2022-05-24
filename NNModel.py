from Layer import *
class NNModel:
    def __init__(self):
        self.layers = []

    def add(self, layer: Layer):
        self.layers.append(layer)
        self.optimizer = None
        self.loss_func = None

    def fit(self, X, y):
        input = X
        output = None
        for layer in self.layers:
            output = layer.forward(input)
            input = output.copy()

        y_hat = self.layers[-1].output
        loss = self.loss(y, y_hat)

        y_hat = np.argmax(y_hat, axis = 1)
        acc = self.accuracy(y, y_hat)

        return acc, loss

    def compile(self, optimizer = 'sgd', loss_func = 'categorical_crossentropy'):
        self.optimizer = optimizer
        self.loss_func = loss_func

    def __CategoricalCrossEntropy(self, y_true, y_hat):
        y_hat_clipped = np.clip(y_hat, 1e-7, 1-1e-7)
        samples = len(y_true)

        if len(y_true.shape) == 1:
            correct_confidences = y_hat_clipped[range(samples), y_true]
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_hat_clipped*y_true, axis = 1)

        neg_log_likelihoods = -np.log(correct_confidences)
        return np.mean(neg_log_likelihoods)

    def loss(self, y_true, y_hat):
        if self.loss_func == 'categorical_crossentropy':
            return self.__CategoricalCrossEntropy(y_true, y_hat)
        else:
            return None

    def accuracy(self, y_true, y_hat):
        return np.mean(y_true == y_hat)