import numpy as np

class NeuralNetwork:

    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.W1 = np.random.randn(self.input_size, self.hidden_size)
        self.b1 = np.zeros((1, self.hidden_size))
        self.W2 = np.random.randn(self.hidden_size, self.output_size)
        self.b2 = np.zeros((1, self.output_size))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        return x * (1 - x)
    
    def forward(self, X):
        self.hidden_output = self.sigmoid(np.dot(X, self.W1) + self.b1)

        self.output = self.sigmoid(np.dot(X, self.W2) + self.b2)
        return self.output
    
    def backward(self, X, y, learning_rate=.25):
        d_output = (y - self.output) * self.sigmoid_derivative(self.output)
        d_W2 = np.dot(self.hidden_output.T, d_output)
        d_b2 = np.sum(d_output, axis=0, keepdims=True)

        d_hidden = np.dot(d_output, self.W2.T) * self.sigmoid_derivative(self.hidden_output)
        d_W1 = np.dot(X.T, d_hidden)
        d_b1 = np.sum(d_hidden, axis=0, keepdims=True)

        self.W2 += learning_rate * d_W2
        self.b2 += learning_rate * d_b2
        self.W1 += learning_rate * d_W1
        self.b1 += learning_rate * d_b1

    def train(self, X, y, epochs=100, learning_rate=.25):
        for epoch in range(epochs):
            output = self.forward(X)

            self.backward(X, y, learning_rate=learning_rate)

            loss = np.mean((y - output) ** 2)

    def predict(self, X):
        return self.forward(X)
