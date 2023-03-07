import numpy as np
from LinearModels import LogisticRegression


if __name__ == "__main__":
    m = LogisticRegression
    X = np.array([[1, 3, 4], [1, -5, 6], [-3, 5, 3]])
    X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
    y = np.array([1, 0, 1])
    preds = np.array([.55, .22, .85])
    grads = LogisticRegression.get_grad(X, y, preds)
    assert np.allclose(grads, np.array([-0.38, 0.22, -3.2, -0.93])), "Что-то не так!"
