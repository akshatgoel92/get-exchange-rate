import os
import math
import torch
import gpytorch
import numpy as np
import src.fetch as fetch
from matplotlib import pyplot as plt



class ExactGPModel(gpytorch.models.ExactGP):
    
    
    def __init__(self, train_x, train_y, likelihood):
        '''
        ---------------------------------
        Initialize a GP model
        ---------------------------------
        '''
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    
    def forward(self, x):
        '''
        ---------------------------------
        Send data forward through the GP
        ---------------------------------
        '''
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def train(train_x, train_y, training_iter, lr):
    '''
    ---------------------------------
    Training loop
    ---------------------------------
    '''
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = ExactGPModel(train_x, train_y, likelihood)

    model.train()
    likelihood.train()
    
    optimizer = torch.optim.Adam([{'params': model.parameters()}, ], lr=lr)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    for i in range(training_iter):
        
        optimizer.zero_grad()
        output = model(train_x)
        
        loss = -mll(output, train_y)
        loss.backward()

        print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
            i + 1, training_iter, loss.item(),
            model.covar_module.base_kernel.lengthscale.item(),
            model.likelihood.noise.item()))

        optimizer.step()

    return(model, likelihood)


def get_test_set(train_x, spot_rate):
    '''
    ----------------------------------
    Get posterior and predictive
    ----------------------------------
    '''
    new_points = torch.autograd.Variable(torch.tensor([spot_rate])).float()
    test_x = torch.cat((train_x, new_points))

    return(test_x)



def predict(model, likelihood, test_x, size=1000):
    '''
    ----------------------------------
    Get posterior and predictive
    ----------------------------------
    '''
    likelihood.eval()
    model.eval()
    
    f_preds = model(test_x)
    y_preds = likelihood(model(test_x))

    f_mean = f_preds.mean
    f_var = f_preds.variance
    f_covar = f_preds.covariance_matrix
    f_samples = f_preds.sample(sample_shape=torch.Size((size,)))

    return(f_preds, y_preds, f_mean, f_var, f_covar, f_samples)


def get_plot(test_x, observed_pred, train_x, train_y, test_x):


    with torch.no_grad():
        # Initialize plot
        f, ax = plt.subplots(1, 1, figsize=(4, 3))
        # Get upper and lower confidence bounds
        lower, upper = observed_pred.confidence_region()
        # Plot training data as black stars
        ax.plot(train_x.numpy(), train_y.numpy(), 'r')
        # Plot predictive means as blue line
        ax.plot(test_x.numpy(), observed_pred.mean.numpy(), 'b')
        # Shade between the lower and upper confidence bounds
        ax.fill_between(test_x.numpy(), lower.numpy(), upper.numpy(), alpha=0.5)
        # Set y-axis delimiters
        ax.set_ylim([60, 110])
        # Set axis legends
        ax.legend(['Observed Data', 'Mean', 'Confidence'])
        # Show
        plt.show()


def main(train_x, train_y, spot_rate=97.91, training_iter=20, lr=0.1):
    '''
    ---------------------------------
    Execute code
    ---------------------------------
    '''
    # Set random seed for reproduciblity
    torch.manual_seed(10)
    # Store trained model and likelihood
    model, likelihood = train(train_x, train_y, training_iter, lr)
    # Construct test set and make predictions
    test_x = get_test_set(train_x, spot_rate)
    predictions = predict(model, likelihood, test_x)
    # Store spot rate predictions
    y_preds = prediction[1]
    # Plot the results
    get_plot(test_x, y_preds, train_x, train_y, test_x)

    return(predictions)