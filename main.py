import numpy as np
from models.linear import LogisticRegression

if __name__ == "__main__":
    m = LogisticRegression.LogisticRegression()
    X = np.array([[1, 3, 4], [1, -5, 6], [-3, 5, 3]])
    X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
    y = np.array([1, 0, 1])
    predictions = np.array([.55, .22, .85])
    grads = m.get_grad(X, y, predictions)
    assert np.allclose(grads, np.array([-0.38, 0.22, -3.2, -0.93])), "Что-то не так!"

    np.random.seed(42)
    m = LogisticRegression.LogisticRegression()
    X = np.random.rand(100, 3)
    y = np.random.randint(0, 1, size=(100,))
    predictions = np.random.rand(100)
    grads = m.get_grad(X, y, predictions)
    assert np.allclose(grads, np.array([23.8698149, 25.27049356, 24.4139452])), "Что-то не так!"
