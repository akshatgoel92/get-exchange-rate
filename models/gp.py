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


def predict(model, likelihood, spot_rate, size=1000):
    '''
    ----------------------------------
    Get posterior and predictive dist.
    ----------------------------------
    '''
    test_x = torch.tensor([spot_rate]).float()
    test_x = torch.autograd.Variable(test_x).float()

    model.eval()
    likelihood.eval()

    f_preds = model(test_x)
    y_preds = likelihood(model(f_preds))

    f_mean = f_preds.mean
    f_var = f_preds.variance
    f_covar = f_preds.covariance_matrix
    f_samples = f_preds.sample(sample_shape=torch.Size((size,)))

    return(f_preds, y_preds, f_mean, f_var, f_covar, f_samples)


def main(df, spot_rate):
    '''
    ---------------------------------
    Execute code
    ---------------------------------
    '''
    torch.manual_seed(10)
    train_x, train_y = get_data(df)
    model, likelihood = train(train_x, train_y)
    predictions = predict(model, likelihood, spot_rate)

    return(**predictions)