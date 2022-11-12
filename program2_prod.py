#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 15 18:22:40 2019

@author: ssingh17
"""

import pandas as pd
import glob
import numpy as np
import warnings
warnings.filterwarnings('ignore')

'''Importing and cleaning USDA corn production estimate data'''

df1 = pd.read_csv('M:Documents/Third Year Paper/Research work/Data/Corn data/Corn_production.csv')
df1 = df1[['Yield Year', 'Month', 'USDA Production Estimates (mb)']]
df1 = df1[(df1['Yield Year'] > 1990)]
month = {'AUG':8, 'SEP':9, 'OCT':10, 'NOV':11, 'JAN':1, 'FEB':2}
df1['Months'] = df1['Month'].map(month)
df1['Year'] = np.where((df1['Month'].str.contains('JAN')) | (df1['Month'].str.contains('FEB')), df1['Yield Year']+1, df1['Yield Year'])

df2 = pd.read_excel('M:Documents/Third Year Paper/Research work/Data/Corn data/WASDE report release date file.xlsx', sheetname = 'Sheet1')
df3 = pd.merge(df1, df2, left_on = ['Year', 'Months'], right_on = ['Year', 'Months'])
df3['Date'] = pd.to_datetime(df3[['Year', 'Months', 'Day']])

df4 = pd.read_excel('M:Documents/Third Year Paper/Research work/Data/Corn data/Analysts Corn estimates.xlsx', sheetname = 'Sheet2')
df5 = pd.merge(df3, df4, how = 'left', left_on = ['Date', 'Yield Year'], right_on = ['Date', 'Analysts Yield Year'])
df5 = df5.sort_values(['Date'], ascending=[0])
df5 = df5[df5['Analysts Production Estimates (mb)'].notnull()]
df5['Unanticipated_prod'] = (np.log(df5['USDA Production Estimates (mb)']/df5['Analysts Production Estimates (mb)']))*100
df5['Date'] = df5['Date'].astype(str)
df5 = df5[['Date', 'Yield Year', 'Months', 'USDA Production Estimates (mb)', 'Analysts Production Estimates (mb)', 'Unanticipated_prod']]
#print(df5[['Date', 'Yield Year']].loc[df5['Analysts Production Estimates (mb)'].isin(['NaN'])])

'''Importing and cleaning Corn future price files (For Feb, Aug, Sep, Oct and Nov events, use December_t futures contract)'''
df_price_1 = pd.DataFrame()
for f in glob.glob('M:Documents/Thesis/Corn Futures Prices/Dec_Corn_Futures/*.csv'):
   df = pd.read_csv(f, skiprows = [0], skipfooter = 1, engine = 'python')
   f_str = str(f)
   abc = (f_str.split('ZCZ', 1)[1]).split('_Barchart', 1)[0]
   if abc > str(79):
       year = '19' + abc
   else:
       year = '20' + abc 
   #print(abc, year)
   df['Contract Year'] = year
   df['Year'] = (df['Date Time'].str.partition('-')[0]).map(lambda x: x.strip())
   df['Month'] = ((df['Date Time'].str.partition('-')[2]).str.partition('-')[0]).map(lambda x: x.strip())
   #df = df[(df['Contract Year'] == df['Year'])]
   df = df[(df['Contract Year'] == df['Year']) & ((df['Month'] == '02') | (df['Month'] == '08') | (df['Month'] == '09') | (df['Month'] == '10') | (df['Month'] == '11'))]
   df_price_1 = df_price_1.append(df, ignore_index=True)

'''For January events, use March_t futures contract'''
df_price_2 = pd.DataFrame()
for f in glob.glob('M:Documents/Thesis/Corn Futures Prices/March_Corn_Futures/*.csv'):
   df = pd.read_csv(f, skiprows = [0], skipfooter = 1, engine = 'python')
   f_str = str(f)
   abc = (f_str.split('ZCH', 1)[1]).split('_Barchart', 1)[0]
   if abc > str(79):
       year = '19' + abc
   else:
       year = '20' + abc 
   #print(abc, year)
   df['Contract Year'] = year
   df['Year'] = (df['Date Time'].str.partition('-')[0]).map(lambda x: x.strip())
   df['Month'] = ((df['Date Time'].str.partition('-')[2]).str.partition('-')[0]).map(lambda x: x.strip())
   #df = df[(df['Contract Year'] == df['Year'])]
   df = df[(df['Contract Year'] == df['Year']) & (df['Month'] == '01')]
   df_price_2 = df_price_2.append(df, ignore_index=True)
  
df_price = df_price_1.append([df_price_2], ignore_index=True)  
df_price.count()
df_price = df_price.rename(columns={'Date Time': 'Date'})
df_price['Year'] = df_price['Year'].astype(int)
df_price['Date'] = df_price['Date'].map(lambda x: x.strip())
df_price = df_price.sort_values(['Date'], ascending=[0])

 
'''Barchart daily futures price data: Open price is recorded on previous day at 7pm CST after 1998, before 1998 Open price recordrd at 8:30 am CST same day, Close price recorded on same day at 1:20 pm CST throughout'''
df_price['Open_next_1'] = df_price['Open'].shift(1)
df_price['Open_prev_1'] = df_price['Open'].shift(-1)
df_price['Open_prev_2'] = df_price['Open'].shift(-2)
df_price['Open_prev_3'] = df_price['Open'].shift(-3)
df_price['Open_prev_4'] = df_price['Open'].shift(-4)
df_price['Close_prev_1'] = df_price['Close'].shift(-1)
df_price['Close_prev_2'] = df_price['Close'].shift(-2)
df_price['Close_prev_3'] = df_price['Close'].shift(-3)
df_price['Close_prev_4'] = df_price['Close'].shift(-4)
df_price['Close_prev_5'] = df_price['Close'].shift(-5)
df_price['Close_prev_6'] = df_price['Close'].shift(-6)
df_price['Close_prev_7'] = df_price['Close'].shift(-7)

df_price['High_next'] = df_price['High'].shift(1)
df_price['High_prev'] = df_price['High'].shift(-1)
df_price['Low_next'] = df_price['Low'].shift(1)
df_price['Low_prev'] = df_price['Low'].shift(-1)


'''Price change calculation'''

df_price['Price_change'] = np.where(df_price['Date'] > '1994-04-30', np.where(df_price['Year'] < 1997, (np.log(df_price['Open']/df_price['Close_prev_1']))*100, (np.log(df_price['Close']/df_price['Open']))*100), (np.log(df_price['Open_next_1']/df_price['Close']))*100)

#df_price['Price_change'] = np.where(df_price['Date'] > '1994-04-30', np.where(df_price['Year'] < str(2013), ((df_price['Close']/df_price['Open']) - 1)*100, ((df_price['Close']/df_price['Open']) - 1)*100), ((df_price['Open_next']/df_price['Close']) - 1)*100)

df_price['Price_change_1'] = np.where(df_price['Date'] > '1994-04-30', np.where(df_price['Year'] < 1997, (np.log(df_price['Open']/df_price['Close_prev_2']))*100, (np.log(df_price['Close']/df_price['Open_prev_1']))*100), (np.log(df_price['Open_next_1']/df_price['Close_prev_1']))*100)


#df_price['Price_change_1'] = np.where(df_price['Date'] > '1994-04-30', np.where(df_price['Year'] < 1997, ((df_price['Open_prev_1']/df_price['Close_prev_2']) - 1)*100, ((df_price['Close_prev_1']/df_price['Open_prev_1']) - 1)*100), ((df_price['Open']/df_price['Close_prev_1']) - 1)*100)

df_price['Price_change_2'] = np.where(df_price['Date'] > '1994-04-30', np.where(df_price['Year'] < 1997, (np.log(df_price['Open']/df_price['Close_prev_3']))*100, (np.log(df_price['Close']/df_price['Open_prev_2']))*100), (np.log(df_price['Open_next_1']/df_price['Close_prev_2']))*100)

df_price['Price_change_3'] = np.where(df_price['Date'] > '1994-04-30', np.where(df_price['Year'] < 1997, (np.log(df_price['Open']/df_price['Close_prev_4']))*100, (np.log(df_price['Close']/df_price['Open_prev_3']))*100), (np.log(df_price['Open_next_1']/df_price['Close_prev_3']))*100)

#df_price['Price_change_prev'] = ((df_price['Close_prev']/df_price['Open_prev']) - 1)*100

df_p = df_price[['Date', 'Year', 'Close', 'Open', 'High', 'Low', 'High_next', 'High_prev', 'Low_next', 'Low_prev', 'Open_next_1', 'Open_prev_1', 'Close_prev_1', 'Close_prev_2', 'Close_prev_3', 'Close_prev_4', 'Close_prev_5', 'Close_prev_6', 'Close_prev_7', 'Price_change', 'Price_change_1', 'Price_change_2', 'Price_change_3']]
df_final = pd.merge(df5, df_p, how = 'inner', on = ['Date'])
df_final = df_final.sort_values(['Date'], ascending=[1])
correlation = df_final['Price_change'].corr(df_final['Unanticipated_prod'])
print('Correlation between price change & Unanticipated production: %.4f' % correlation)
#print(df_final[['Date', 'Yield Year']].loc[df_final['Price_change'].isin(['NaN'])])
df_final.to_csv('M:Documents/Third Year Paper/Research work/data_withProd_final.csv', index = False)


