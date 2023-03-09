import numpy as np


def batch_generator(X, y, batch_size):
    """

    :param random_state: RandomState instance
    :param X: np.array[n_objects, k_features]
    :param y: np.array[n_objects_target]
    :param batch_size: batch size of data
    Batch generator with random sampling of batch_size elements from the sample
    """
    np.random.seed(42)
    X = np.array(X)
    y = np.array(y)
    perm = np.random.permutation(len(X))

    epochs = range(0, int(np.floor(X.size // batch_size)))
    for batch_start in epochs:
        left_, right_ = batch_start * batch_size, (batch_start + 1) * batch_size
        X_batch, y_batch = X[perm][left_:right_], y[perm][left_:right_]
        yield X_batch, y_batch
