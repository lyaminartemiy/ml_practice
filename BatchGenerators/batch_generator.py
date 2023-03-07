import numpy as np


def batch_generator(X, y, batch_size):
    """

    :param X: np.array[n_objects, k_features]
    :param y: np.array[n_objects_target]
    :param batch_size: batch size of data
    Batch generator with random sampling of batch_size elements from the sample
    """
    np.random.seed(42)
    perm = np.random.permutation(X)

    epochs = int(np.floor(X.shape[0] // batch_size))
    for batch_ in range(0, epochs):
        X_batch, y_batch = X[perm][(batch_ * batch_size):((batch_ + 1) * batch_size)], \
                           y[perm][(batch_ * batch_size):((batch_ + 1) * batch_size)]
        X, y = X[perm][batch_size:], y[perm][batch_size:]
        yield X_batch, y_batch