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
    print(X)
    # perm = np.random.permutation(X).astype(int)
    np.random.shuffle(X)

    epochs = int(np.floor(X.shape[0] // batch_size))
    for batch_ in range(0, epochs):
        left_, right_ = batch_ * batch_size, (batch_ + 1) * batch_size
        X_batch, y_batch = X[left_:right_, :], X[left_:right_, :]
        yield X_batch, y_batch
