import scipy
from collections import Counter
import numpy as np


def euclidean(point, data):
    return np.sqrt(np.sum((point-data)**2, axis=1))

def most_common(lst):
    return max(set(lst), key=lst.count)

class KNearestNeighbor:
    def __init__(self, k, dist_metric=euclidean):
        self.k = k
        self.dist_metric = dist_metric
    
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        neighbors = []
        for x in X_test:
            distances = self.dist_metric(x, self.X_train)
            y_sorted = [y for _, y in sorted(zip(distances, self.y_train))]
            neighbors.append(y_sorted[:self.k])
        return list(map(most_common, neighbors))
    
    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)
        accuracy = sum(y_pred == y_test) / len(y_test)
        return accuracy


###
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

