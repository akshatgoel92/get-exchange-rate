# Imports
from pylab import mpl, plt
import pandas as pd 
import numpy as np 
import requests
import json


# Plot settings
plt.style.use('seaborn')
mpl.rcParams['font.family'] = 'serif'



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



def summarize(df):
	'''
	------------------
	Make a line graph
	------------------
	'''
	print(df.describe())



def parse_data(response):
    '''
    ----------------------
    Parse the API response
    ----------------------
    '''
    df = pd.DataFrame(json.loads(response.text)['rates']).T
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


def plot(df):
	'''
	------------------
	Make a line graph
	------------------
	'''
	# Basic figure
	plt.figure(figsize = (10, 4))
	plt.plot(df['INR'], linewidth = 1)
	plt.plot(df['sma'], linewidth = 1, label = "SMA")
	plt.plot(df['lma'], linewidth = 1, label = "LMA")
	
	# Annotations
	plt.title("GBP-INR Exchange rate")
	plt.xlabel("Date")
	plt.ylabel("Time")
	plt.legend(loc=0)
	plt.show()

	return(df)



def main():
	'''
	------------------
	Execution goes here
	------------------
	'''
	args = {'start': '2009-01-01', 'end': '2020-07-31',
			'target': 'INR', 'base': 'GBP'} 
	
	response = get_data(**args)
	df = parse_data(response)
	df = get_moving_avg(df)
	
	summarize(df)
	plot(df)
	
	

if __name__ == '__main__':

	main()