# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 22:07:02 2020

@author: ssingh17
"""

import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import matplotlib.dates as mdates
import numpy as np
import warnings
warnings.filterwarnings('ignore')

'''Figure 1 (x-axis labeling to year distorts the plot slightly, attempt to correct later)'''
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


'''Figure 2 (x-axis labeling to year distorts the plot slightly, attempt to correct later)'''
df = pd.read_csv('M:Documents/Third Year Paper/Research work/data_withProd_final.csv')
df['Date'] =  pd.to_datetime(df['Date'], format='%Y-%m-%d')
df.set_index(df['Date'], inplace = True)
plt.plot(df['Date'], df['Unanticipated_prod'], label='_nolegend_') 
#plt.plot(df['Date'], df['Price_change'], label='_nolegend_')  
plt.xlabel('Date') 
plt.ylabel('Market Surprise (%)') 
plt.legend() 
plt.show()








fig, ax = plt.subplots()
ax.plot(df['Date'], df['Volatility'])
ax.plot(df['Date'], df['Volatility_prev'])
#ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
fig.autofmt_xdate()
plt.show()



#set ticks every week
ax.xaxis.set_major_locator(mdates.YearLocator(5))
#set major ticks format
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

datemin = np.datetime64(df['Date'][0], 'Y')
datemax = np.datetime64(df['Date'][-1], 'Y') + np.timedelta64(1, 'Y')
ax.set_xlim(datemin, datemax)
fig.autofmt_xdate()
plt.show()






x = mdates.date2num(df['Date'])
plt.plot_date(x, df['Volatility'], fmt="r-")
plt.show()

datemin = datetime.date(df['Date'].min().year, 1, 1)
datemax = datetime.date(df['Date'].max().year + 1, 1, 1)
ax.set_xlim(datemin, datemax)



plt.show()




years = mdates.YearLocator() 
ax.xaxis.set_major_locator(years)
#ax.xaxis.set_major_locator(mdates.DayLocator(interval=30))
fig.autofmt_xdate()
plt.show()





x = [datetime.strptime(d, "%Y-%m-%d") for d in df['Date']] 
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=50))
plt.plot(x, df['Volatility'].values, label = 'On Event Day') 
plt.plot(x, df['Volatility_prev'].values, label = 'Day before Event Day')  
#years = mdates.YearLocator()
#ax.xaxis.set_major_locator(years)
df.dtypes
