# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 22:07:02 2020

@author: ssingh17
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings('ignore')


'''Figure 1'''
df = pd.read_csv('M:Documents/Third Year Paper/Research work/data_withProd_final.csv')

df['High_prev_2'] = df['High'].shift(-2)
df['Low_prev_2'] = df['Low'].shift(-2)

df['Volatility'] = np.where(df['Date'] > '1994-04-30', np.where(df['Year'] < 1997, (np.log(df['High']/df['Low']))*100, (np.log(df['High']/df['Low']))*100), (np.log(df['High_next']/df['Low_next']))*100)

df['Volatility_prev'] = np.where(df['Date'] > '1994-04-30', np.where(df['Year'] < 1997, (np.log(df['High_prev']/df['Low_prev']))*100, (np.log(df['High_prev']/df['Low_prev']))*100), (np.log(df['High']/df['Low']))*100)

df['Volatility_prev_2'] = np.where(df['Date'] > '1994-04-30', np.where(df['Year'] < 1997, (np.log(df['High_prev_2']/df['Low_prev_2']))*100, (np.log(df['High_prev_2']/df['Low_prev_2']))*100), (np.log(df['High_prev']/df['Low_prev']))*100)

df['Date'] =  pd.to_datetime(df['Date'], format='%Y-%m-%d')
df.set_index(df['Date'], inplace = True)
plt.plot(df['Date'], df['Volatility'], label = 'On Event Day') 
plt.plot(df['Date'], df['Volatility_prev'], label = 'Day before Event Day') 
plt.plot(df['Date'], df['Volatility_prev_2'], label = '2 Days before Event Day') 
plt.xlabel('Date') 
plt.ylabel('Volatility (%)') 
plt.legend() 
plt.show()


'''Figure 2'''
df = pd.read_csv('M:Documents/Third Year Paper/Research work/data_withProd_final.csv')
df['Date'] =  pd.to_datetime(df['Date'], format='%Y-%m-%d')
df.set_index(df['Date'], inplace = True)
plt.plot(df['Date'], df['Unanticipated_prod'], label='_nolegend_') 
#plt.plot(df['Date'], df['Price_change'], label='_nolegend_')  
plt.xlabel('Date') 
plt.ylabel('Market Surprise (%)') 
plt.legend() 
plt.show()


