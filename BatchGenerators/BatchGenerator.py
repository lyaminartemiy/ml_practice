import numpy as np


def batch_generator(X, y, batch_size):
    perm = np.random.permutation(X)

    epochs = int(np.floor(X.shape[0] // batch_size))
    for batch_ in range(0, epochs):
        X_batch, y_batch = X[perm][batch_:((batch_ + 1) * batch_size)], \
                           y[perm][batch_:((batch_ + 1) * batch_size)]
        X, y = X[perm][batch_size:], y[perm][batch_size:]
        yield X_batch, y_batch
