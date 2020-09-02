
import numpy as np
from matplotlib import pyplot as plt

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C




def get_data():
    '''
    ===========================
    Input: 
    Output:
    ===========================
    '''
    # Now the noisy case
    X = np.linspace(0.1, 9.9, 20)
    X = np.atleast_2d(X).T

    # Observations and noise
    y = f(X).ravel()
    dy = 0.5 + 1.0 * np.random.random(y.shape)
    
    noise = np.random.normal(0, dy)
    y += noise

    return(X, y)


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


def get_prediction(gp):
    '''
    ===========================
    Input: 
    Output:
    ===========================
    ''' 
    # Make the prediction on the meshed x-axis (ask for MSE as well)
    return(gp.predict(x, return_std=True))


def get_plot():
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
    alpha = dy ** 2

    X, y = get_data()
    gp = get_estimator(kernel, alpha)
    gp_fit = fit_estimator(X, y, gp)
    
    y_pred, sigma = get_prediction(gp)
    get_plot(x, y_pred, sigma)


if __name__ == '__main__':
    main()