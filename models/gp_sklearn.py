
import numpy as np

from src import fetch
from matplotlib import pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel as wn


def f(x):
    """
    ===========================
    The function to predict
    ===========================
    """
    return x * np.sin(x)


def get_estimator(kernel, alpha=0.0, n_restarts_optimizer=0):
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
    return(gp.predict(X, return_cov=True))


def get_plot(X, x, y, y_mean, y_cov):
    '''
    ===========================
    Input: 
    Output:
    ===========================
    '''
    # Plot the function, the prediction and the 95% confidence interval based on # the MSE
    plt.plot(X, y_mean, 'k', lw=3, zorder=9)
    plt.fill_between(X, y_mean - np.sqrt(np.diag(y_cov)),
                     y_mean + np.sqrt(np.diag(y_cov)),
                     alpha=0.5, color='k')
    plt.scatter(X[:, 0], y, c='r', s=50, zorder=10, edgecolors=(0, 0, 0))
    plt.title("Initial: %s\nOptimum: %s\nLog-Marginal-Likelihood: %s"
              % (kernel, gp.kernel_,
                 gp.log_marginal_likelihood(gp.kernel_.theta)))
    plt.tight_layout()


def main():
    '''
    ===========================
    Input: 
    Output:
    ===========================
    '''
    _, _, X, x, y = fetch.main()
    kernel = 1.0 * RBF(length_scale=100.0, length_scale_bounds=(1e-2, 1e3)) + \
             wn(noise_level=1, noise_level_bounds=(1e-10, 1e+1))
    
    gp = get_estimator(kernel)
    gp_fit = fit_estimator(X, y, gp)
    
    y_mean, y_cov = get_prediction(x, gp_fit)

    get_plot(X, x, y, y_mean, y_cov)


if __name__ == '__main__':
    main()