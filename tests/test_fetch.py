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

        self.response = fetch.get_data(**args)
        self.df = fetch.parse_data(self.response)
        
        self.df = fetch.get_moving_avg(self.df)
        self.sum = fetch.summarize(self.df)
        self.fig = ui.get_plot(self.df)

    def test_data_success(self):
        self.assertEqual(self.response.status_code, 200)

    def test_df_shape(self):
        self.assertEqual(self.df.shape, (2982, 3))

    def test_sum_shape(self):
        self.assertEqual(self.sum.shape, (8, 3))

    def test_fig_type(self):
        self.assertEqual(type(self.fig), go.Figure)

    def test_data(self):
        self.assertEqual(type(self.fig), go.Figure)

    def test_gp_data(self):
        pass
    

if __name__ == '__main__':
    unittest.main()