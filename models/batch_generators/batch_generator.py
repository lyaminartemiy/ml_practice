import numpy as np


def batch_generator(X, y, batch_size, random_state):
    """

    :param random_state: RandomState instance
    :param X: np.array[n_objects, k_features]
    :param y: np.array[n_objects_target]
    :param batch_size: batch size of data
    Batch generator with random sampling of batch_size elements from the sample
    """
    np.random.seed(random_state)
    perm = np.random.permutation(X).astype(int)

    epochs = int(np.floor(X.shape[0] // batch_size))
    for batch_start in range(epochs):
        left_, right_ = batch_start * batch_size, (batch_start + 1) * batch_size
        X_batch, y_batch = X[perm][left_:right_], y[perm][left_:right_]
        yield X_batch, y_batch
