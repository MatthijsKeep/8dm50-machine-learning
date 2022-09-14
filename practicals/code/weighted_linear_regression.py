import numpy as np

def w_lsq(X,y,d):
    """
    Weighted least squares linear regression
    :param X: Input data matrix
    :param y: Target vector
    :param d: Weight vector for input data
    :return: Estimated coefficient vector for the linear regression
    """

    ones = np.ones((len(X), 1)) # add column of ones for the intercept
    X = np.concatenate((ones, X), axis=1)
    W = np.diag(np.full(len(X), d))  # Create weight matrix (with weights on diagonal)

    return np.linalg.multi_dot([np.linalg.inv(np.linalg.multi_dot([X.T, W, X])), X.T, W, y])

