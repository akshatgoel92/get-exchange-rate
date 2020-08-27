# Imports
import plotly.graph_objects as go
import pandas as pd 
import numpy as np 
import requests
import json


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
	sum = df.describe()
	return(sum)



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


def get_plot(df):
	'''
	------------------
	Add moving avgs
	------------------
	'''
	fig = go.Figure()
	
	fig.add_trace(go.Scatter(x=df.index, y=df.INR,
							 mode='lines',
							 name='Spot Rate'))
	
	fig.add_trace(go.Scatter(x=df.index, y=df.sma,
							 mode='lines',
							 name='Short-term avg.'))
	
	fig.add_trace(go.Scatter(x=df.index, y=df.lma,
							 mode='lines', 
							 name='Long-term avg.'))

	fig.update_layout(title='GBP INR Over Time',
					  xaxis_title='Date',
					  yaxis_title='INR')

	return(fig)
	

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
	df = parse_data(response)
	
	df = get_moving_avg(df)
	sum = summarize(df)
	fig = get_plot(df)

	return(sum, fig)
	

if __name__ == '__main__':

	main()