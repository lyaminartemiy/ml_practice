import numpy as np
from BatchGenerators.BatchGenerator import batch_generator


def logit(X, w):
    """

    :param X: np.array([n_objects, n_features])
    :param w: np.array([weights])
    :return: vector of scalar products of weights and features
    """
    return np.dot(X, w)


def sigmoid(p):
    """

    :param p: np.array([predictions])
    :return: vector of sigmoid values
    """
    return 1 / (1 + np.exp(-p))


class LogisticRegression(object):

    def __init__(self):
        self.w = None

    def fit(self, X, y, lr=0.1, batch_size=10):
        n_objects = X.shape[0]
        self.w = np.array(np.random.rand(n_objects))

        X = np.c_[X, np.ones(X.shape[0])]

        loses = []

        epochs = int(np.floor(X.shape[0] // batch_size))
        for i in range(0, epochs):
            for X_batch, y_batch in batch_generator(X, y, batch_size):

                predictions = self.predict_proba_internal(X_batch)

                loss = self.__loss(y_batch, predictions)
                loses.append(loss)

                self.w = self.w - (lr * self.get_grad(X_batch, y_batch, predictions))

        return loses

    def get_grad(self, X_batch, y_batch, predictions):
        grad = np.transpose(X_batch) @ (predictions - y_batch)
        return grad

    def predict_proba(self, X):
        X = np.c_[X, np.ones(X.shape[0])]
        return sigmoid(logit(X, self.w))

    def predict_proba_internal(self, X):
        return sigmoid(logit(X, self.w))

    def predict(self, X, threshold=0.5):
        return self.predict_proba(X) >= threshold

    def get_weights(self):
        return self.w.copy()

    @staticmethod
    def __loss(y, p):
        return -np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))
