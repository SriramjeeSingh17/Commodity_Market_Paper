#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 15 18:12:16 2019

@author: ssingh17
"""

import pandas as pd
import numpy as np
import statsmodels.formula.api as sm
from termcolor import colored

import warnings
warnings.filterwarnings('ignore')

'''Regression Model - Type 1'''
data_final = pd.read_csv('M:/Documents/Third Year Paper/Research work/data_withProd_final.csv')
#data_final.count()
#data_final = data_final[data_final['Year'] >= 2000]
#data_final = data_final[data_final['Months'] >= 8]
result=sm.ols(formula='Price_change ~ Unanticipated_prod', data = data_final).fit()
print(result.params)
result.summary()


'''Regression Model - Report Leakage'''

data_final['Price_change_leak_1f'] = np.where(data_final['Date'] > '1994-04-30', np.where(data_final['Year'] < 1997, (np.log(data_final['Open_prev_1']/data_final['Close_prev_5']))*100, (np.log(data_final['Close_prev_1']/data_final['Close_prev_5']))*100), (np.log(data_final['Open']/data_final['Close_prev_5']))*100)

data_final['Price_change_leak_1'] = (np.log(data_final['Close_prev_1']/data_final['Close_prev_5']))*100
#data_final['Price_change_leak_1o'] = (np.log(data_final['Open_prev_1']/data_final['Close_prev_5']))*100
data_final['Price_change_leak_1a'] = (np.log(data_final['Close_prev_1']/data_final['Close_prev_6']))*100
data_final['Price_change_leak_1b'] = (np.log(data_final['Close_prev_1']/data_final['Close_prev_4']))*100

data_final['Price_change_leak_2'] = (np.log(data_final['Close_prev_2']/data_final['Close_prev_6']))*100
data_final['Price_change_leak_2a'] = (np.log(data_final['Close_prev_2']/data_final['Close_prev_7']))*100
data_final['Price_change_leak_2b'] = (np.log(data_final['Close_prev_2']/data_final['Close_prev_5']))*100


data_final = data_final[data_final['Year'] < 2005]
#data_final = data_final[data_final['Months'] >= 8]
result=sm.ols(formula='Price_change_leak_1b ~ Unanticipated_prod', data = data_final).fit()
print(colored(result.params, 'red', attrs = ['bold']))
#print(result.params)
result.summary()

