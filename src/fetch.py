# Imports
import pandas as pd 
import numpy as np
import requests
import json

# Import torch
import gpytorch
import torch





def get_data(start, end, target, base):
	'''
	----------------------------
	Get request for time series
	----------------------------
	'''
	args = 'start_at={}&end_at={}&symbols={}&base={}'
	url = 'https://api.exchangeratesapi.io/history?' + args
	response = requests.get(url.format(start, end, target, base))

	return(response)


def parse_data(df_string):
	'''
	----------------------
	Parse the API response
	----------------------
	'''
	df = pd.DataFrame(json.loads(df_string)['rates']).T
	df.index = pd.to_datetime(df.index)
	df = df.sort_index()

	return(df)



def get_moving_avg(df, sma=30, lma=200):
	'''
	------------------
	Add moving avgs
	------------------
	'''
	df['sma'] = df['INR'].rolling(window=sma).mean()
	df['lma'] = df['INR'].rolling(window=lma).mean()

	return(df)



def get_long_dates(df):
	'''
	------------------
	Add moving avgs
	------------------
	'''
	df['long'] = (df['sma'] > df['lma']).astype(int) 

	return(df)



def summarize_data(df):
	'''
	------------------
	Make a line graph
	------------------
	'''
	sum = df.describe()
	
	return(sum)



def get_gp_torch_data(df):
    '''
    ________________________
    Convert data to tensors
    ________________________
    '''
    # Prepare the training data
    train_x = torch.from_numpy(np.array(df['INR'].shift(1)[1:])).float()
    train_y = torch.from_numpy(np.array(df['INR'][:-1])).float()

    # Wrap the training data in a variable object
    train_x = torch.autograd.Variable(train_x).float()
    train_y = torch.autograd.Variable(train_y).float()

    return(train_x, train_y)


def get_gp_sk_data(df, lag=1):
    '''
    ===========================
    Input: 
    Output:
    ===========================
    '''
    # Now the noisy case
    X = np.array(df['INR'].shift(1)[lag:])
    X = np.atleast_2d(X).T
    
    # Make a mesh for visualization
    x = np.atleast_2d(np.linspace(0, 5, 150)).T

    # Observations and noise
    y = np.array(df['INR'][:-lag])

    return(X, x, y)



def main():
	'''
	------------------
	Execution goes here
	------------------
	'''
	args = {'start': '2009-01-01', 'end': '2020-08-26', 
			'target': 'INR', 
			'base': 'GBP'}

	response = get_data(**args)
	df = parse_data(response.text)
	
	df = get_moving_avg(df)
	sum = summarize_data(df)
	X, x, y = get_gp_sk_data(df)

	return(sum, df, X, x, y)
	

if __name__ == '__main__':

	main()