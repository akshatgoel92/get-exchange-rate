# Import packages
import argparse


def get_fd_value(principal, rate, period):
    '''
    ---------------
    Execute the function calls.
    ---------------
    '''
    return((principal*(1 - rate**period))/(1-rate))


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
    
    args = parser.parse_args()
    
    rate = args.rate
    period = args.period
    principal = args.principal
    
    return(rate, period, principal)

    
def main():
    '''
    ---------------
    Execute the function calls.
    ---------------
    '''
    # Get the arguments
    rate, period, principal = parse_args()
    amount = get_fd_value(principal, rate, period)
    print('The total final value of this FD investment is {}'.format(amount))

    return


if __name__ == '__main__':
    '''
    ---------------
    Call main function.
    ---------------
    '''
    main()