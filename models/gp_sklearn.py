
import numpy as np

from src import fetch
from matplotlib import pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C


def f(x):
    """The function to predict."""
    return x * np.sin(x)



def get_estimator(kernel, alpha, n_restarts_optimizer=10):
    '''
    ===========================
    Input: 
    Output:
    ===========================
    '''
    # Instantiate a Gaussian Process model
    gp = GaussianProcessRegressor(kernel=kernel, 
                                  alpha=alpha,
                                  n_restarts_optimizer=n_restarts_optimizer)

    return(gp)


def fit_estimator(X, y, gp): 
    '''
    ===========================
    Input: 
    Output:
    ===========================
    '''
    # Fit to data using Maximum Likelihood Estimation of the parameters
    return(gp.fit(X, y))


def get_prediction(X, gp):
    '''
    ===========================
    Input: 
    Output:
    ===========================
    ''' 
    # Make the prediction on the meshed x-axis (ask for MSE as well)
    return(gp.predict(X, return_std=True))


def get_plot(X, x, y, dy, y_pred, sigma):
    '''
    ===========================
    Input: 
    Output:
    ===========================
    '''
    # Plot the function, the prediction and the 95% confidence interval based on # the MSE
    plt.figure()
    
    plt.plot(x, f(x), 'r:', label=r'$f(x) = x\,\sin(x)$')
    plt.errorbar(X.ravel(), y, dy, fmt='r.', markersize=10, label='Observations')
    plt.plot(x, y_pred, 'b-', label='Prediction')
    plt.fill(np.concatenate([x, x[::-1]]),
             np.concatenate([y_pred - 1.9600 * sigma,
                            (y_pred + 1.9600 * sigma)[::-1]]),
             alpha=.5, fc='b', ec='None', label='95% confidence interval')
    
    plt.xlabel('$x$')
    plt.ylabel('$f(x)$')
    plt.ylim(-10, 20)
    
    plt.legend(loc='upper left')
    plt.show()


def main():
    '''
    ===========================
    Input: 
    Output:
    ===========================
    '''
    kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
    X, x, y, dy = fetch.get_data()
    alpha = dy ** 2

    gp = get_estimator(kernel, alpha)
    gp_fit = fit_estimator(X, y, gp)
    
    y_pred, sigma = get_prediction(x, gp_fit)
    get_plot(X, x, y, dy, y_pred, sigma)


if __name__ == '__main__':
    main()