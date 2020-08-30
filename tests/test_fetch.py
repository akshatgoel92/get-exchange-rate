
import unittest
import src.ui as ui
import src.fetch as fetch
from unittest.mock import Mock, patch

import plotly.graph_objects as go
import pandas as pd
import numpy as np
import gpytorch
import torch




class TestDataFetch(unittest.TestCase):


    def setUp(self):
        
        self.args = {'start': '2009-01-01', 
                     'end': '2020-08-26', 
                     'target': 'INR', 
                     'base': 'GBP'}

        self.df_string = {''}
        
        
    def test_get_data(self):
        
        response = fetch.get_data(**self.args)
        assertEqual(response.status, 200)
        
    
    def test_df_string(self):
        
        response = fetch.get_data(**self.args)
        self.df_string = response.text
        assertEqual(1, 1)
    
    
    def test_parse_data(self):
        
        self.df = fetch.parse_data(self.df_string)

        self.assertEqual(type(self.df), pd.DataFrame)
        self.assertEqual(self.df.shape, (2982, 3))

        
    def test_sum_shape(self):
        
        self.assertEqual(self.sum.shape, (8, 3))

    def test_gp_data(self):
        
        self.assertEqual(type(self.gp_data, tuple)
        self.assertEqual(type(self.gp_data[0]), torch.tensor)
        self.assertEqual(type(self.gp_data[1]), torch.tensor)
    

if __name__ == '__main__':
    unittest.main()