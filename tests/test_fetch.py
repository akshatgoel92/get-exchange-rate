
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

        self.df_string = '''{"rates": {"2011-11-21":{"INR":81.8511627907},
                                       "2013-01-31":{"INR":84.1540256709}},
                             "start_at":"2009-01-01", "base":"GBP", 
                             "end_at":"2020-08-26"}'''
        
        
    def test_get_data(self):
        
        response = fetch.get_data(**self.args)
        self.assertEqual(response.status_code, 200)
        
    
    def test_parse_data(self):
        
        self.df = fetch.parse_data(self.df_string)
        self.assertEqual(type(self.df), pd.DataFrame)
        

    def test_summarize_data(self):

        self.df = fetch.summarize_data(self.df_string)
        self.assertEqual(self.sum.shape, (8, 3))

    
    def test_sum_shape(self):
        pass
        

    
    def test_gp_data(self):
        
        self.assertEqual(type(self.gp_data, tuple))
        self.assertEqual(len(self.gp_data), 2)
        self.assertEqual(type(self.gp_data[0]), torch.tensor)
        self.assertEqual(type(self.gp_data[1]), torch.tensor)
    

if __name__ == '__main__':
    unittest.main()