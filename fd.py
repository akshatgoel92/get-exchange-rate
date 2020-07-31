# Import packages
import argparse
import numpy as np


def get_one_time_fd_value(principal, rate, period):
    '''
    ---------------
    Execute the function calls.
    ---------------
    '''
    return(principal*(rate**period))


def get_recurring_fd_value(principal, rate, period):
    '''
    ---------------
    Execute the function calls.
    ---------------
    '''
    amounts = [get_one_time_fd_value(principal, rate, tenure) for tenure in range(period + 1)]
    return(np.sum(np.array(amounts)))    

def get_amount(amount, rate, period):
    '''
    ---------------
    Execute the function calls.
    ---------------
    '''
    principal = amount/(rate**period)
    return(principal)
    
def parse_args():
    '''
    ---------------
    Parse command line arguments here.
    ---------------
    '''
    parser = argparse.ArgumentParser(description='Process inputs to exchange rate routine.')
    parser.add_argument('--rate', type = float, help = 'Assumed FD rate of interest.', required = False)
    parser.add_argument('--period', type = int, help = 'No. of periods FD will be held for.', required = False)
    parser.add_argument('--principal', type = int, help = 'The total initial investment in rupees.', required = False)
    parser.add_argument('--r', dest='accumulate', action='store_const', const=get_recurring_fd_value, 
                        default=get_one_time_fd_value, help='sum the integers (default: find the max)')
    parser.add_argument('--a', dest='accumulate', action='store_const', const=get_amount, 
                        default=get_one_time_fd_value, help='sum the integers (default: find the max)')
    
    args = parser.parse_args()
    
    rate = args.rate
    period = args.period
    principal = args.principal
    recurring = args.accumulate
    
    print(args)
    
    return(rate, period, principal, recurring)


def main():
    '''
    ---------------
    Execute the function calls.
    ---------------
    '''
    msg = 'The total final value of this FD investment is {}'
    rate, period, principal, recurring = parse_args()
    amount = recurring(principal, rate, period)
    print(msg.format(amount))

    return


if __name__ == '__main__':
    '''
    ---------------
    Call main function.
    ---------------
    '''
    main()