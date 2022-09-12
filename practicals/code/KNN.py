import scipy
from collections import Counter
import numpy as np

class KNN:
    """
    Class to do KNN classification with
    """
    def __init__(self, k):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def distance(self, X1, X2):
        distance = scipy.spatial.distance.euclidean(X1, X2)
        return distance

    def predict(self, X_test):
        final_output = []
        for i in range(len(X_test)):
            d = []
            votes = []
            for j in range(len(self.X_train)):
                dist = scipy.spatial.distance.euclidean(self.X_train[j] , X_test[i])
                d.append([dist, j])
            d.sort()
            d = d[0:self.k]
            for d, j in d:
                votes.append(self.y_train[j])
            ans = Counter(votes).most_common(1)[0][0]
            final_output.append(ans)
            
        return final_output

    def score(self, X_test, y_test):
        predictions = self.predict(X_test)
        return (predictions == y_test).sum() / len(y_test)


def KNN_regressor(X_train, X_test, y_train, y_test, n_neigh):
    """
    KNN regression function
    :param X_train: train data
    :param X_test: test data
    :param y_train: train targets
    :param y_test: test targets
    :param n_neigh: k value for nearest neighbor algorithm
    :return: Estimated coefficient vector for the linear regression
    
    """
    y_pred = np.zeros(y_test.shape)

    for row in range(len(X_test)):
        euclidian_distance = np.sqrt(np.sum((X_train - X_test[row])**2, axis = 1 ))
        y_pred[row] = y_train[np.argsort(euclidian_distance, axis = 0)[:n_neigh]].mean()
        
    #Finding the root mean squared error 
    RMSE = np.sqrt(np.mean((y_test - y_pred)**2))

    return RMSE

