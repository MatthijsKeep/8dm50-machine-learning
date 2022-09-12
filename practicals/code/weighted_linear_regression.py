import numpy as np

def w_lsq(X,y,d):
    """
    Weighted least squares linear regression
    :param X: Input data matrix
    :param y: Target vector
    :param d: Weight vector for input data
    :return: Estimated coefficient vector for the linear regression
    """

    # # add column of ones for the intercept
    ones = np.ones((len(X), 1))
    X = np.concatenate((ones, X), axis=1)

    # # calculate the coefficients including weight vector

    # Create weight matrix (with weights on diagonal)
    W = np.diag(np.full(len(X), d))

    # calculate dot products with weight matrix W
    A = np.linalg.inv(np.linalg.multi_dot([np.transpose(X), W, X]))
    B = np.linalg.multi_dot([np.transpose(X), W, y])
    
    # calculate coefficients
    beta = np.dot(A, B)

    # The operations above in 1 go
    #beta = np.dot(np.linalg.inv(np.linalg.multi_dot([X.T, W, X])), (np.linalg.multi_dot([X.T, W, y])))
    
    return beta