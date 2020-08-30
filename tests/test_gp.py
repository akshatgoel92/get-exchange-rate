import unittest
import model.gp_test as gp

import plotly.graph_objects as go
import pandas as pd
import numpy as np



class TestExactGP(unittest.TestCase):


    def setUp(self):
        
        # Set seed
        torch.manual_seed(10391)
        # Set training iterations
        self.training_iter = 50 
        # Set learning rate
        self.lr = 0.1
        # Training data is 100 points in [0,1] inclusive regularly spaced
        self.train_x = torch.linspace(0, 1, 100)
        # True function is sin(2*pi*x) with Gaussian noise
        self.train_y = torch.sin(train_x * (2 * math.pi)) + torch.randn(train_x.size()) * math.sqrt(0.04)
        # Train model and get likelihood
        self.model, self.likelihood = gp.train()


    def test_train_model_type(self):
        self.assertEqual(type(gp.train(**self.train_args)), 200)

    def test_train_likelihood_type(self):
        self.assertEqual(type(gp.train(**self.train_args), 200)

    def test_train_df_shape(self)
        self.assertEqual(gp.train(**self.train_args), 200)
    
    def tearDown(self):

        del self.train_x
        del self.train_y


if __name__ == '__main__':
    unittest.main()