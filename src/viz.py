
# Import packages
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt 

# Set plot options
plt.style.use('seaborn') 
mpl.rcParams['font.family'] = 'serif'



def sim_data():
	'''
	-------------------------------------
	Input: None
	Output: Simulated data for line plot 
	-------------------------------------
	'''
	y = np.random.standard_normal(200)
	x = np.arange(len(y))

	return(x, y)


def sim_data_2d():
	'''
	-------------------------------------
	Input: None
	Output: Simulated 2D data  
	-------------------------------------
	'''
	y = np.random.\
		standard_normal((200, 2)).cumsum(axis = 0)

	return(y)

def rescale_2d(y):
	'''
	-------------------------------------
	Input: 2-d dataset
	Output: 2-d dataset rescaled 
	-------------------------------------
	'''
	y[:, 0] = y[:, 0] * 100

	return(y)



def line_plot(x, y):
	'''
	------------------
	Input: X and Y series 
	Output: Line plot
	------------------
	'''
	title = "Simulated time series"
	xlab = "Simulated time"
	ylab = "Simulated values"
	plt.plot(x, y)
	
	plt.title(title)
	plt.xlabel(xlab)
	
	plt.ylabel(ylab)
	plt.show()



def line_plot_scaled(y):
	'''
	------------------
	Input: X and Y series 
	Output: Line plot
	------------------
	'''
	# Set figure size
	plt.figure(figsize = (10, 6))
	
	# First subplot
	plt.subplot(211)
	plt.plot(y[:, 0], lw = 1.5,
			 label = '1st')

	plt.plot(y[:, 0])
	plt.legend(loc=0)

	# Second subplot
	plt.subplot(212)
	plt.plot(y[:, 1], lw = 1.5, 
			 label = '2nd')
	plt.legend(loc = 9)

	# Show
	plt.show()



def main():
	'''
	------------------
	Input: X and Y series 
	Output: Line plot
	------------------
	'''
	# Setting the seed
	np.random.seed(1000)
	
	# Line plot for 1-d
	x, y = sim_data()
	line_plot(x, y)

	# Line plot for 2-d
	y = sim_data_2d()
	line_plot(x, y)

	y = rescale_2d(y)
	line_plot_scaled(y)


if __name__ == '__main__':
	'''
	----------------------
	Execution lives here
	----------------------
	'''
	main()
