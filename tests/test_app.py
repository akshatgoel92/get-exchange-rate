import models.gp as gp
import pandas as pd
import numpy as np
import unittest
import gpytorch
import torch
import math


class TestApp(unittest.TestCase):


    def setUp():
        '''
        ----------------------
        Input: 
        Output: 
        ----------------------
        '''


    def test_integration_gp(self):
        '''
        ----------------------
        Input: 
        Output: 
        ----------------------
        '''
        try: 
            gp.main(self.df)
        except Exception as e:
            self.fail(e)


if __name__ == '__main__':
    unittest.main()