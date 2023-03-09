import numpy as np
from models.batch_generators.batch_generator import batch_generator


def logit(x, w):
    """

    :param x: np.array([n_objects, n_features])
    :param w: np.array([weights])
    :return: vector of scalar products of weights and features
    """
    return np.dot(x, w)


def sigmoid(p):
    """

    :param p: np.array([predictions])
    :return: vector of sigmoid values
    """
    return 1. / (1 + np.exp(-p))


class LogisticRegression(object):

    def __init__(self):
        self.w = None

    def fit(self, X, y, epochs=10, lr=0.1, batch_size=100):

        n_objects, k_features = X.shape

        if self.w is None:
            np.random.seed(42)
            self.w = np.random.randn(k_features + 1)

        X = np.concatenate((np.ones((n_objects, 1)), X), axis=1)

        loses = []

        for i in range(epochs):
            for X_batch, y_batch in batch_generator(X, y, batch_size):

                predictions = self.predict_proba_internal(X_batch)

                loss = self.__loss(y_batch, predictions)
                loses.append(loss)

                self.w = self.w - lr * self.get_grad(X_batch, y_batch, predictions)

        return loses

    @staticmethod
    def get_grad(X_batch, y_batch, predictions):
        grad = np.transpose(X_batch) @ (predictions - y_batch)
        return grad

    def predict_proba(self, X):
        n_objects, k_features = X.shape
        X = np.concatenate((np.ones((n_objects, 1)), X), axis=1)
        return sigmoid(logit(X, self.w))

    def predict_proba_internal(self, X):
        return sigmoid(logit(X, self.w))

    def predict(self, X, threshold=0.5):
        return self.predict_proba(X) >= threshold

    def get_weights(self):
        return self.w.copy()

    @staticmethod
    def __loss(y, p):
        p = np.clip(p, 1e-10, 1 - 1e-10)
        return -np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))
