import numpy as np
from models.batch_generators.batch_generator import batch_generator


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

    def __init__(self, lr=0.1, batch_size=10, random_state=42):
        self.w = None
        self.lr = lr
        self.batch_size = batch_size
        self.random_state = random_state

    def fit(self, X, y):
        n_objects = X.shape[0]

        if self.w is None:
            self.w = np.array(np.random.rand(n_objects))

        X = np.c_[np.ones(n_objects), X]

        loses = []

        epochs = int(np.floor(X.shape[0] // self.batch_size))
        for i in range(0, epochs):
            for X_batch, y_batch in batch_generator(X, y, self.batch_size, self.random_state):
                predictions = self.predict_proba_internal(X_batch)

                loss = self.__loss(y_batch, predictions)
                loses.append(loss)

                self.w = self.w - (self.lr * self.get_grad(X_batch, y_batch, predictions))

        return loses

    @staticmethod
    def get_grad(X_batch, y_batch, predictions):
        grad = np.transpose(X_batch) @ (predictions - y_batch)
        return grad

    def predict_proba(self, X):
        X = np.c_[np.ones(X.shape[0]), X]
        return sigmoid(logit(X, self.w))

    def predict_proba_internal(self, X):
        return sigmoid(logit(X, self.w))

    def predict(self, X, threshold=0.5):
        return (self.predict_proba_internal(X) >= threshold).astype(int)

    def get_weights(self):
        return self.w.copy()

    @staticmethod
    def __loss(y, p):
        return -np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))
