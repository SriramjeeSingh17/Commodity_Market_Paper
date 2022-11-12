#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 15 18:12:16 2019

@author: ssingh17
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import statsmodels.formula.api as sm
import sys
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

len(data_final)
data_final['Months'].unique()

















stdoutOrigin=sys.stdout 
sys.stdout = open(r'M:\Documents\Third Year Paper\Research work\Results\leakage_results.txt', 'w+')
result=sm.ols(formula='Price_change_leak_1 ~ Unanticipated_prod', data = data_final).fit()
print('Price_change_leak_1 (y): price change of Close at day t-1 wrt day day t-5')
print(result.summary())
result=sm.ols(formula='Price_change_leak_1a ~ Unanticipated_prod', data = data_final).fit()
print('Price_change_leak_1a (y): price change of Close at day t-1 wrt day day t-6')
print(result.summary())
result=sm.ols(formula='Price_change_leak_1b ~ Unanticipated_prod', data = data_final).fit()
print('Price_change_leak_1b (y): price change of Close at day t-1 wrt day day t-4')
print(result.summary())

result=sm.ols(formula='Price_change_leak_2 ~ Unanticipated_prod', data = data_final).fit()
print('Price_change_leak_2 (y): price change of Close at day t-2 wrt day day t-6')
print(result.summary())
result=sm.ols(formula='Price_change_leak_2a ~ Unanticipated_prod', data = data_final).fit()
print('Price_change_leak_2a (y): price change of Close at day t-2 wrt day day t-7')
print(result.summary())
result=sm.ols(formula='Price_change_leak_2b ~ Unanticipated_prod', data = data_final).fit()
print('Price_change_leak_2b (y): price change of Close at day t-2 wrt day day t-5')
print(result.summary())
sys.stdout.close()
sys.stdout=stdoutOrigin 


'Locally weighted Linear Regression'
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
len(data_final)
data_final['Unanticipated_prod'].min()

def wm(point, X, tau):
    m = X.shape[0]
    w = np.mat(np.eye(m)) 
    for i in range(m): 
        xi = X[i] 
        d = (-2 * tau * tau) 
        w[i, i] = np.exp(np.dot((xi-point), (xi-point).T)/d) 
        
    return w


def predict(X, y, point, tau):
    m = X.shape[0]  
    X = X.reshape(m, 1)
    X_ = np.concatenate((X, np.ones(X.shape)), axis=1) 
    point_ = np.array([point, 1])  
    w = wm(point_, X_train, tau) 
    theta = np.linalg.pinv(X_.T*(w * X_))*(X_.T*(w * y)) 
    pred = np.dot(point_, theta) 
    return theta, pred


def plot_predictions(X, y, tau, nval):
    #X_test = np.linspace(-3, 3, nval) 
    preds = []
    for point in X_test: 
        theta, pred = predict(X, y, point, tau) 
        preds.append(pred)
    X_test = np.array(X_test).reshape(len(X_test),1)
    preds = np.array(preds).reshape(len(X_test),1)
    plt.plot(X, y, 'b.')
    plt.plot(X_test, preds, 'r.') 
    plt.show()
    



X=data_final['Unanticipated_prod'].values  
X_train = X_train.reshape(X_train.shape[0], 1)
y = data_final['Price_change'].values
y_train = y_train.reshape(y_train.shape[0], 1)    
plot_predictions(X_train, y_train, 0.08, len(X_test))

predict(X_train, y_train, point, tau)


for point in X_test:
    print(point)



X_test[0:5]

'''Regression Model - Type 2'''
data_final = pd.read_csv('M:/Documents/Third Year Paper/Research work/data_withProd_final.csv')
df_new = data_final[['Date', 'Yield Year', 'Unanticipated_prod', 'Price_change']] 
#df_new = df_new[(df_new['Yield Year'] >= 2007)]
df_new_train = df_new[(df_new['Yield Year'] < 2017)]
df_new_test = df_new[(df_new['Yield Year'] >= 2017)]

X_train, X_test, y_train, y_test = df_new_train.iloc[:, 2:3], df_new_test.iloc[:, 2:3], df_new_train.iloc[:, -1:], df_new_test.iloc[:, -1:]

sc = MinMaxScaler(feature_range=(0, 1))
X_train, y_train, X_test, y_test = sc.fit_transform(X_train), sc.fit_transform(y_train), sc.fit_transform(X_test), sc.fit_transform(y_test) 

model=LinearRegression()
model.fit(X_train,y_train)
print('Intercept:', model.intercept_)
print('Coefficient:', model.coef_)

y_pred = model.predict(X_test)
inv_y_pred = sc.inverse_transform(y_pred)
inv_y_test = sc.inverse_transform(y_test)
abc = np.concatenate((inv_y_test, inv_y_pred), axis=1)
mse = mean_squared_error(abc[:,0],abc[:,1])
print('MSE: %6f' % mse)
df_y = pd.DataFrame({'Log_Return_on_Price_test':abc[:,0],'Log_Return_on_Price_predicted':abc[:,1]})
df_y

df_y['Sign Compare'] = np.where(df_y['Log_Return_on_Price_test'] != 0, np.where(np.sign(df_y['Log_Return_on_Price_test']) == np.sign(df_y['Log_Return_on_Price_predicted']), 'Yes', 'No'), 'Yes')
df_y.groupby('Sign Compare').count()





