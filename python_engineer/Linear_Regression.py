import numpy as np
class LinearRegression:
    def __init__(self, lr=3e-4, n_iters=1000):
        self.lr = lr
        self.n_iters = 1000
        self.weights = None
        self.bias = None

    def fit(self, X, Y):
        # init parameters
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iters):
            y_predicted = X @ self.weights + self.bias

            dw = (1 / self.n_samples) * 2 * np.sum(X.T @ (y_predicted - Y))
            db = (1 / self.n_samples) * 2 * np.sum((y_predicted - Y))

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        y_predicted = X @ self.weights + self.bias
        return y_predicted

