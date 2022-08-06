import numpy as np

class NeuralNetwork():

    def __init__(self, input_size, hidden_size, output_size):
        self.epochs = 1000
        self.learning_rate = 1e-3

        self.W1 = np.random.randn(hidden_size, input_size) * 0.01
        self.b1 = np.zeros((hidden_size, 1))
        self.W2 = np.random.randn(output_size, hidden_size) * 0.01
        self.b2 = np.zeros((output_size, 1))

    def sigmoid(x):
        return 1/(1 + np.exp(-x))

    def forward(self, X):
        self.hidden_activations = self.sigmoid(X @ self.W1 + self.b1)
        self.output_layer = self.hidden_activations @ self.W2 + self.b2
        return self.sigmoid(self.output_layer)
    
    def MeanSquaredError(label, prediction):
        return (np.mean((label - prediction) ** 2))

    def train(self, inputs, labels):
        for i in range(self.epochs):
            logits = self.forward(inputs)

            J = self.MeanSquaredError(labels, logits)

