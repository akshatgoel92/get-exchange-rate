import unittest
import src.ui as ui
import src.fetch as fetch

import plotly.graph_objects as go
import pandas as pd
import numpy as np



class TestDataFetch(unittest.TestCase):


    def setUp(self):
        
        args = {'start': '2009-01-01', 'end': '2020-08-26', 
                'target': 'INR', 'base': 'GBP'}

        self.sum, self.df = src.fetch.main(**args)
        self.fig = ui.get_plot()

    def test_fig_type(self):
        self.assertEqual(type(self.fig), go.Figure)
    

if __name__ == '__main__':
    unittest.main()