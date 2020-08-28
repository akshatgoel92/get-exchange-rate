import os
import math
import torch
import gpytorch
from matplotlib import pyplot as plt


# We will use the simplest form of GP model, exact inference
class ExactGPModel(gpytorch.models.ExactGP):
    
    
    def __init__(self, train_x, train_y, likelihood):
        '''
        ___________________
        Forward propogation
        > Inputs:
        > Outputs: 
        _____
        '''
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    
    def forward(self, x):
        '''
        ___________________
        Forward propogation
        > Inputs:
        > Outputs: 
        ___________________
        '''
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def get_data():
    '''
    ___________________
    Forward propogation
    > Inputs:
    > Outputs: 
    ___________________
    '''
    # Training data is 100 points in [0,1] inclusive regularly spaced
    train_x = torch.linspace(0, 1, 100)
    # True function is sin(2*pi*x) with Gaussian noise
    train_y = torch.sin(train_x * (2 * math.pi)) + torch.randn(train_x.size()) * math.sqrt(0.04)

    return(train_x, train_y)


def train(train_x, train_y, training_iter=100):
    '''
    ___________________
    Forward propogation
    > Inputs:
    > Outputs: 
    ___________________
    '''
    # Initialize likelihood and model
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = ExactGPModel(train_x, train_y, likelihood)

    # Find optimal model hyperparameters
    model.train()
    likelihood.train()

    # Use the adam optimizer
    optimizer = torch.optim.Adam([{'params': model.parameters()}, ], lr=0.1)

    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    for i in range(training_iter):
        
        # Zero gradients from previous iteration
        optimizer.zero_grad()
        # Output from model
        output = model(train_x)
        
        # Calculate loss and backprop gradients
        loss = -mll(output, train_y)
        loss.backward()
        
        print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
            i + 1, training_iter, loss.item(),
            model.covar_module.base_kernel.lengthscale.item(),
            model.likelihood.noise.item()))
        
        optimizer.step()

    return(model, likelihood)


def predict(model, likelihood, test_x):
    '''
    ___________________
    Forward propogation
    > Inputs:
    > Outputs: 
    ___________________
    '''
    f_preds = model(test_x)
    y_preds = likelihood(model(test_x))

    f_mean = f_preds.mean
    f_var = f_preds.variance
    
    f_covar = f_preds.covariance_matrix
    f_samples = f_preds.sample(sample_shape=torch.Size(1000,))

    return(f_preds, y_preds, f_mean, f_var, f_covar, f_samples)


def main():
    '''
    ___________________
    Forward propogation
    > Inputs:
    > Outputs: 
    ___________________
    '''
    train_x, train_y = get_data()
    model, likelihood = train(train_x, train_y)


if __name__ == '__main__':
    main()
