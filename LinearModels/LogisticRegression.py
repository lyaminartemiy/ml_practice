import numpy as np


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

    @staticmethod
    def __loss(y, p):
        """

        :param y: target
        :param p: predictions
        :return:
        """
        return -np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))
