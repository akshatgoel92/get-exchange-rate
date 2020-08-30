import models.gp as gp
import pandas as pd
import numpy as np
import unittest
import gpytorch
import torch
import math

from gpytorch.likelihoods.gaussian_likelihood import GaussianLikelihood


class TestExactGP(unittest.TestCase):


    def setUp(self):
        '''
        ----------------------
        Input: 
        Output: 
        ----------------------
        '''
        # Set seed
        torch.manual_seed(10391)
        # Set training iterations
        self.training_iter = 50 
        # Set learning rate
        self.lr = 0.1
        # Training data is 100 points in [0,1] inclusive regularly spaced
        self.train_x = torch.linspace(0, 1, 100)
        # True function is sin(2*pi*x) with Gaussian noise
        self.train_y = torch.sin(self.train_x * (2 * math.pi)) + torch.randn(self.train_x.size()) * math.sqrt(0.04)
        # Train model and get likelihood
        self.model, self.likelihood = gp.train(self.train_x, self.train_y, self.training_iter, self.lr)


    def test_train_model_type(self):
        '''
        ----------------------
        Input: 
        Output: 
        ----------------------
        '''
        self.assertEqual(type(self.model), gp.ExactGPModel)

    
    def test_train_likelihood_type(self):
        '''
        ----------------------
        Input: 
        Output: 
        ----------------------
        '''
        self.assertEqual(type(self.likelihood), GaussianLikelihood)

if __name__ == '__main__':
    unittest.main()