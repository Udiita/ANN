
import numpy as np

class Perceptron:

    def __init__(self, num_features, learning_rate):
        self.weights = np.random.rand(num_features)
        self.learning_rate = learning_rate

    def fit(self, X, y, n_epochs=100):

        n_samples = X.shape[0]
        n_features = X.shape[1]

        # Add column of 1s
        X = np.concatenate([np.ones((n_samples, 1)), X], axis=1)

        for i in range(n_epochs):
            for j in range(n_samples):
                if y[j]*np.dot(self.weights, X[j, :]) <= 0:
                    self.weights += self.learning_rate * 2 * y[j] * X[j, :]

    def predict(self, X):
        n_samples = X.shape[0]
        X = np.concatenate([np.ones((n_samples, 1)), X], axis=1)
        y = np.matmul(X, self.weights)
        y = np.vectorize(lambda val: 1 if val > 0 else -1)(y)
        return y

## Copy from appendix A
X = np.array([[-0.6508, 0.1097, 4.009], [-0.6508, 0.1097, 4.009], [-0.6508, 0.1097, 4.009], [-0.6508, 0.1097, 4.009]])
Y = np.array([1, 1, 1, 1])

p = Perceptron(4, 0.01)
p.fit(X, Y)
print(p.weights)
test_X = np.array([[-0.6508, 0.1097, 4.009], [-0.6508, 0.1097, 4.009], [-0.6508, 0.1097, 4.009]])
print(p.predict(test_X))
