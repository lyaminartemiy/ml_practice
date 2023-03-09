import numpy as np
from models.linear.LogisticRegression import LogisticRegression


class RidgeRegression(LogisticRegression):

    def __init__(self, alpha=1, lr=0.1, batch_size=10, random_state=42):
        super().__init__(lr, batch_size, random_state)
        self.alpha = alpha
        self.w = None

    def get_grad(self, X_batch, y_batch, predictions):
        grad = np.transpose(X_batch) @ (predictions - y_batch)
        grad_l2 = 2 * self.alpha * self.w
        grad_l2[0] = 0
        return grad + grad_l2

    def __loss(self, y, p):
        return -np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)) + self.alpha * (self.w ** 2)
