"""
File to make cross validation easier, and a bit more readable.



"""

from os import sync
import numpy as np
import matplotlib.pyplot as plt

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.linear_model import LinearRegression

def generate_dataset(n=100, degree=1, noise=1, factors=None):
    """
    Generate a dataset with a polynomial function.

    Parameters
    ----------
    n : int
        Number of samples to generate.
    degree : int
        Degree of the polynomial function.
    noise : float
        Standard deviation of the noise.
    factors : list
        List of factors for the polynomial function.
    
    Returns
    -------
    X : array
        Array of shape (n, 1) containing the input values.
    y : array
        Array of shape (n, 1) containing the output values.
    """

    
    np.random.seed(42) 
    X = np.random.uniform(low=-1, high=1, size=n)

    if factors is None:
        factors = np.random.uniform(0, 10, degree+1)
    

    y = np.zeros(X.shape)
    for idx in range(degree+1):
        y += factors[idx] * (X ** idx)

    # add noise
    y += np.random.normal(-noise, noise, n)

    order = np.argsort(X)
    X = np.array(X[order])
    y = np.array(y[order])

    return X, y


def PolynomialRegression(degree=2, **kwargs):
    """
    Helper function to create a polynomial regression model.

    Parameters
    ----------
    degree : int
        Degree of the polynomial function.
    **kwargs : dict
        Keyword arguments to pass to the underlying model.

    Returns
    -------
    model : Pipeline
        A polynomial regression model.
    
    """

    return make_pipeline(PolynomialFeatures(degree), LinearRegression(**kwargs))


def crossval_gridsearch(synthetic=False, synthetic_params=None, dataset=None):
    """
    Perform cross validation on a dataset using GridSearchCV.

    Parameters
    ----------
    synthetic : bool
        Whether to use a synthetic dataset or not.
    synthetic_params : dict
        Parameters to use for the synthetic dataset.
    dataset : string
        Name of the dataset to use. (if synthetic is False)
    
    Returns
    -------
    model : GridSearchCV
        The model that was trained.
    
    """

    
    if synthetic == False and dataset is None:
        raise ValueError("Please provide a dataset")
    if synthetic and dataset is not None:
        raise ValueError("Please provide either a dataset or synthetic data")
    
    if synthetic:
        if synthetic_params is None:
            synthetic_params = [100, 4, 1, 1, 1]
        X_train, y_train = generate_dataset(n=synthetic_params[0], degree=synthetic_params[1], noise=synthetic_params[2])
        X_test, y_test = generate_dataset(n=synthetic_params[0], degree=synthetic_params[1], noise=synthetic_params[3]*3)

    if dataset == 'diabetes':
        from sklearn import datasets

        # load the diabetes dataset
        diabetes = datasets.load_diabetes()

        # use only one feature
        X = diabetes.data[:, np.newaxis, 2]
        y = diabetes.target

        # print the value counts of the target

        # print(f"Value counts of the target: {np.unique(y, return_counts=True)}")

        # split the data into training/testing sets
        X_train = X[:-20]
        X_test = X[-20:]

        # split the targets into training/testing sets
        y_train = y[:-20]
        y_test = y[-20:]

        # print the value counts of y_train and y_test

        




    
    param_grid = {'polynomialfeatures__degree': np.arange(21)[1:]}

    poly_grid = GridSearchCV(PolynomialRegression(), param_grid, 
                             cv=10, # 20x cross validation 
                             scoring='neg_root_mean_squared_error', # RMSE as performance metric
                             verbose=0) # Do not print intermediate results 

    poly_grid.fit(X_train.reshape(-1, 1), y_train)

    print(f"Best estimator is: {poly_grid.best_estimator_}")
    print(f"Best cross-validation score: {poly_grid.best_score_:.2f}")
    print(f"Test-set score: {poly_grid.score(X_test.reshape(-1, 1), y_test):.2f}")

    y_pred = poly_grid.predict(X_test.reshape(-1,1)) 
    print(y_pred[:3], y_test[:3])

    plt.figure(figsize=(10, 6))
    plt.title(f"Polynomial regression of degree {poly_grid.best_estimator_[0]}", size=16)
    plt.scatter(X_train.reshape(-1, 1), y_train)
    plt.plot(X_test.reshape(-1, 1), y_pred, c="red")

    # Plot test set in orange
    plt.scatter(X_test.reshape(-1, 1), y_test, c='orange')
    plt.show()

    # Print test result beneath plot
    print(f"Score of best estimator on training data: {poly_grid.score(X_train.reshape(-1,1), y_train):.3f}")
    print(f"Score on the separately generated test data: {poly_grid.score(X_test.reshape(-1,1), y_test):.3f}")

    # Plot validation results (avg test score on CV validation set) 
    CI_up = poly_grid.cv_results_['mean_test_score'] + (1.96 * poly_grid.cv_results_['std_test_score'] / np.sqrt(poly_grid.cv))
    CI_down = poly_grid.cv_results_['mean_test_score'] - (1.96 * poly_grid.cv_results_['std_test_score'] / np.sqrt(poly_grid.cv))

    fig, ax = plt.subplots(figsize=(10,6))
    x = np.arange(21)[1:]
    ax.plot(x, poly_grid.cv_results_['mean_test_score'])
    ax.fill_between(x, CI_up, CI_down, color='b', alpha=0.15)
    ax.set_ylim(ymin=-100, ymax=0)
    ax.set_xlabel("Polynomial degree")
    ax.set_ylabel("negative RMSE of model")
    ax.set_title("CV performance with CI");

    return poly_grid
    

# below for testing purposes 

if __name__ == "__main__":
    crossval_gridsearch(synthetic=True, synthetic_params=[100, 4, 1, 1, 1])









