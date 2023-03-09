import numpy as np
from models.linear import LogisticRegression

if __name__ == "__main__":

    m = LogisticRegression.LogisticRegression()

    X = np.array([[1, 3, 4], [1, -10, 6], [-3, 5, 3]])
    y = np.array([1, 0, 1])

    loses = m.fit(X, y)

    X_test = np.array([[1, 3, 4.5], [1, -10, 6], [-3, 500, 3.9]])

    predictions = m.predict(X_test)
    print(predictions)
