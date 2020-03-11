# Import packages
import pandas as pd
import argparse
import requests
import json


def get_data(base = 'GBP', target = 'INR'):
    '''
    -----------
    Make GET request to foreign exchange API for given
    base and target currencies.
    -----------
    '''
    url = 'https://api.exchangeratesapi.io/latest?base={}&symbols={}'
    response = requests.get(url.format(base, target))
                            
    return(response)


def parse_data(response):
    '''
    -------------
    Parse the API response to get float exchange rate.
    -------------
    '''
    # Parse the API response
    rate = json.loads(response.text)['rates']['INR']
    
    return(rate)


def get_fees(exchange_rate, base_fees):
    '''
    --------------
    Inputs: Exchange rate
    Inputs: Fees in base currency
    Output: Total tuition fees in target currency
    --------------
    '''
    total_fees = exchange_rate*base_fees
    return(total_fees)
     

def parse_args():
    '''
    ---------------
    Parse command line arguments here.
    ---------------
    '''
    parser = argparse.ArgumentParser(description='Process inputs to exchange rate routine.')
    
    parser.add_argument('base', type = str, default = 'GBP', help = 'The base currency three letter code.')
    parser.add_argument('target', type = str, default= 'INR', help = 'The target currency three letter code.')
    parser.add_argument('tuition', type = int, default = 30400, help = 'The total tuition fees in base currency')
    
    args = parser.parse_args()
    
    base = args.base
    target = args.target
    tuition = args.tuition
    
    return(base, target, tuition)

    
def main():
    '''
    ---------------
    Execute the function calls.
    ---------------
    '''
    # Get the arguments
    base, target, tuition = parse_args()
    
    # Get the response
    response = get_data(base, target)
    print(response.text)
    
    # Parse the data properly
    rate = parse_data(response)
    print('The current {}-{} exchange rate is {}.'.format(base, target, rate))
    
    # Calculate the total fees
    fees = get_fees(rate, fees)
    print('The total fees at this rate is {}.'.format(fees))
    
    return


if __name__ == '__main__':
    '''
    ---------------
    Call main function.
    ---------------
    '''
    main()
    
