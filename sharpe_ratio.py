"""
Created on Sun Sep 15 20:24:01 2019

@author: unalunsal

Optimize a portfolio by using Sharpe-ratio.                                                                                                                                                                                  
                                                                                                                                                                         
"""                                                                                                                                                                                                                                                                                            
import pandas as pd                                                                                                                                                                               
import matplotlib.pyplot as plt                                                                                                                                                                                   
import numpy as np                                                                                                                                                                                
import datetime as dt                                                                                                                                                                             
from util import get_data, plot_data       # CURRENTLY, MAKING SURE THAT THIS COMPONENT IS ACCURATE. These are helper functions that needs to be updated.                                                                                                                                                                       
from scipy.optimize import minimize        # I will post similar version of util.py soon. Thank you. 
                                                                                                                                                                                  
                                                                                                                                                                            
def optimize_portfolio(sd=dt.datetime(2008,1,1), ed=dt.datetime(2009,1,1), \
    syms=['GOOG','AAPL','GLD','XOM'], gen_plot=False):                                                                                                                                                                                    
                                                                                                                                                                                  
    # Read in adjusted closing prices for given symbols, date range                                                                                                                                                                               
    dates = pd.date_range(sd, ed)                                                                                                                                                                                 
    prices_all = get_data(syms, dates)  # automatically adds SPY                                                                                                                                                                                  
    prices = prices_all[syms]  # only portfolio symbols                                                                                                                                                                                   
    prices_SPY = prices_all['SPY']  # only SPY, for comparison later                                                                                                                                                                              
    prices_diff = (prices / prices.shift(1)) - 1  # get daily return of each stock 

    def optimum_values(port_allocation, risk_free):
        port_allocation = np.array(port_allocation)
        port_return = np.sum(prices_diff.mean() * port_allocation) * 252
        port_volatility = np.sqrt(np.dot(port_allocation.T,np.dot(prices_diff.cov()*252,port_allocation)))
        port_sharpe = (port_return - risk_free)/port_volatility 
        return np.array([port_return,port_volatility,port_sharpe])

    def negative_val(port_allocation):
        return optimum_values(port_allocation, 0)[2] * -1                                

    # check for the short positions later. 
    def constraint_sum(port_allocation):
        return np.sum(port_allocation) -1 

    constraint_val = ({'type':'eq','fun':constraint_sum})   # create constraint variable   
    constraint_bound = ((0,1),)*prices_diff.shape[1]        # create as many tuples / boundaries as the number of stocks
    initial_guess = np.full(prices_diff.shape[1], 1/prices_diff.shape[1]).tolist()  # as many initial guess as the number of stocks      

    optimum_results = minimize(negative_val, initial_guess, method='SLSQP', bounds=constraint_bound, constraints=constraint_val)
                          
    # note that the values here ARE NOT meant to be correct for a test case                                                                                                                                                                               
    allocs = optimum_results.x   # allocations
    
    cr = np.dot((prices.ix[0:1,:] / prices.ix[-1,:]), allocs) - 1           # Cumulative return          
    cr = cr.tolist()[0]
    adr = np.average(np.dot(prices_diff.dropna(), allocs))  # average of portfolio daily return 
    sddr = np.std(np.dot(prices_diff.dropna(), allocs))     # standard deviation of portfolio daily return                       
    sr = optimum_values(allocs, 0)[2]                       # sharpe ratio
    
    # Get daily portfolio value                                           
    port_val = prices * allocs    
    ret = port_val.sum(axis = 1) # daily portfolio value.
                                                                                                                                                  
    # Compare daily portfolio value with SPY using a normalized plot                                                                                                                                                                              
    if gen_plot == True:                                                                                                                                                                          
        plt.plot(prices_SPY / prices_SPY[0])
        plt.plot(ret / ret[0])
        plt.ylabel('Price')
        plt.xlabel('Date')
        plt.legend(('SPY','Portfolio'))
        plt.grid(True, 'major','y',ls = '--')
        plt.savefig('project2.jpg')
        pass
                                                                                              
                                                                                                                                                                                  
    return allocs, cr, adr, sddr, sr                                                                                                                                                                              
                                                                                                                                                                                  
def test_code():
    start_date = dt.datetime(2008,1,1)                                                                                                                                                                                    
    end_date = dt.datetime(2009,1,1)                                                                                                                                                                              
    symbols = ['GOOG', 'AAPL', 'GLD', 'XOM']     # You can call any stock price by using its symbol.                                                                                                                                                                         
                                                                                                                                                                                  
    # Assess the portfolio                                                                                                                                                                                
    allocations, cr, adr, sddr, sr = optimize_portfolio(sd = start_date, ed = end_date,\
        syms = symbols, \
        gen_plot = False)                                                                                                                                                                                 
                                                                                                                                                                                  
    # Print statistics                                                                                                                                                                                    
    print(f"Start Date: {start_date}")                                                                                                                                                                                    
    print(f"End Date: {end_date}")                                                                                                                                                                                
    print(f"Symbols: {symbols}")                                                                                                                                                                                  
    print(f"Allocations:{allocations}")                                                                                                                                                                                   
    print(f"Sharpe Ratio: {sr}")                                                                                                                                                                                  
    print(f"Volatility (stdev of daily returns): {sddr}")                                                                                                                                                                                 
    print(f"Average Daily Return: {adr}")                                                                                                                                                                                 
    print(f"Cumulative Return: {cr}")                                                                                                                                                                             
                                                                                                                                                                                  
if __name__ == "__main__":                                                                                                                                                                                
                                                                                                                                                                              
    test_code()        
