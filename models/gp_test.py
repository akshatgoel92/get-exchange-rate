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
        ______________________
        Initialize a GP model
        ______________________
        '''
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    
    def forward(self, x):
        '''
        _________________________________
        Send data forward through the GP
        _________________________________
        '''
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def get_data(df):
    '''
    ________________________
    Convert data to tensors
    ________________________
    '''
    # Prepare the training data
    train_x = torch.from_numpy(np.array(df['INR'].shift(1)[1:])).float()
    train_y = torch.from_numpy(np.array(df['INR'][:-1])).float()

    # Wrap the training data in a variable object
    train_x = torch.autograd.Variable(train_x).float()
    train_y = torch.autograd.Variable(train_y).float()

    return(train_x, train_y)


def train(train_x, train_y, training_iter, lr):
    '''
    ___________________
    Training loop
    ___________________
    '''
    # Initialize likelihood
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    # Initialize model
    model = ExactGPModel(train_x, train_y, likelihood)
    # Find optimal model hyperparameters
    model.train()
    # Find optimal likelihood parameters
    likelihood.train()
    # Use the Adam optimizer
    optimizer = torch.optim.Adam([{'params': model.parameters()}, ], lr=lr)
    # "Loss" for GPs: the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    # Enter the training loop
    for i in range(training_iter):
        # Zero gradients from previous iteration
        optimizer.zero_grad()
        # Output from model
        output = model(train_x)
        # Calculate loss
        loss = -mll(output, train_y)
        # Calculate graduents on the loss function
        loss.backward()
        # Print progress of the model
        print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
            i + 1, training_iter, loss.item(),
            model.covar_module.base_kernel.lengthscale.item(),
            model.likelihood.noise.item()))
        # One calculation step
        optimizer.step()

    return(model, likelihood)


def predict(model, likelihood, spot_rate, size=1000):
    '''
    ___________________________
    Get posterior + predictive
    ___________________________
    '''
    # Create a tensor from the spot rate scalar
    test_x = torch.tensor([spot_rate]).float()
    # Wrap the test data in a Variable object
    test_x = torch.autograd.Variable(test_x).float()
    # Put model into evaluation mode
    model.eval()
    # Put likelihood into evaluation mode
    likelihood.eval()
    # Get posterior distribution
    f_preds = model(test_x)
    # Get predictive distribution
    y_preds = likelihood(model(f_preds))
    # Get posterior mean
    f_mean = f_preds.mean
    # Get posterior variance
    f_var = f_preds.variance
    # Get posterior covariance
    f_covar = f_preds.covariance_matrix
    # Get samples from the posterior
    f_samples = f_preds.sample(sample_shape=torch.Size((size,)))
    # Return statement
    return(f_preds, y_preds, f_mean, f_var, f_covar, f_samples)


def main(df, spot_rate):
    '''
    ___________________
    Execute code
    ___________________
    '''
    # Set seed for reproducible results
    torch.manual_seed(10)
    # Get the training data
    train_x, train_y = get_data(df)
    # Get the model and likelihood objects from training
    model, likelihood = train(train_x, train_y)
    # Get the results we need
    f_preds, y_preds, f_mean, f_var, f_covar, f_samples = predict(model, likelihood, spot_rate)