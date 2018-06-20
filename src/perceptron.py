import numpy as np


class Perceptron(object):

    def __init__(self,
                 eta=0.01,
                 n_iter=50,
                 random_state=1):
        self.eta = eta
        """
        Learning rate (between 0.0 and 1.0)
        :type eta: float
        """

        self.n_iter = n_iter
        """
        Number of passes over the training dataset.
        :type n_iter: int
        """

        self.random_state = random_state
        """
        Random number generator seed for random weight initialization
        :type random_state: int
        """

    def fit(self, X, y):
        """
        Fit training data to target values
        :param X: Training vectors with n_samples of n_features
        :type X: np.array, shape=[n_samples, n_features]
        :param y: Target values
        :type y: np.array, shape=[n_samples]
        :return: self
        :rtpye: object
        """
        rgen = np.random.RandomState(self.random_state)
        self._weights = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])

        self.errors = []

        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self._weights[1:] += update * xi
                self._weights[0] += update # bias unit
                errors += int(update != 0.0)
            self.errors.append(errors)
        return self

    def net_input(self, X):
        """
        Calculate net input
        """
        return np.dot(X, self._weights[1:]) + self._weights[0]

    def predict(self, X):
        """
        Return class label after unit step
        """
        return np.where(self.net_input(X) >= 0.0, 1, -1)