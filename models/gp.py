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



def get_predict_datasets(new_points, data=None):
    '''
    ----------------------------------
    Get posterior and predictive
    ----------------------------------
    '''
    # Convert new points to a Tensor
    new_tensor = torch.autograd.Variable(torch.tensor(new_points)).float()
    # Concatenate
    full_data = torch.cat((data, new_tensor)) if data is not None else None

    return(new_tensor, full_data)



def predict(model, likelihood, test_x, size):
    '''
    ----------------------------------
    Get posterior and predictive
    ----------------------------------
    '''
    model.eval()
    likelihood.eval()
    
    f_preds = model(test_x)
    f_mean = f_preds.mean
    f_var = f_preds.variance
    f_covar = f_preds.covariance_matrix
    f_samples = f_preds.sample(sample_shape=torch.Size((size,)))

    y_preds = likelihood(f_preds)
    y_upper, y_lower = y_preds.confidence_region()

    return(f_preds, f_mean, f_var, f_covar, f_samples, y_preds, y_upper, y_lower)



def get_plot(observed_pred, y_preds, y_upper, y_lower):


    with torch.no_grad():
        # Initialize plot
        f, ax = plt.subplots(1, 1, figsize=(4, 3))
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


def main(train_x, train_y, 
         spot_rate=[97.91], upper_bound = [100], 
         lower_bound = [90], training_iter=10, lr=0.1, size=1000):
    '''
    ---------------------------------
    Execute code
    ---------------------------------
    '''
    # Set random seed for reproduciblity
    torch.manual_seed(10)
    # Store trained model and likelihood
    model, likelihood = train(train_x, train_y, training_iter, lr)
    # Construct test set
    test_x, _ = get_predict_datasets(spot_rate)
    # Make predictions
    predictions = predict(model, likelihood, test_x, size)
    # Construct datasets for plotting: posterior mean
    mean_line = get_predict_datasets(spot_rate, train_x)
    # Posterior upper bound
    upper_bound = get_predict_datasets(upper_bound, train_x)
    # Poterior lower bound 
    lower_bound = get_predict_datasets(lower_bound, train_x)
    # Store spot rate predictions
    y_preds = predictions[1]
    # Plot the results
    get_plot(y_preds, train_x, train_y, test_x)

    return(predictions)