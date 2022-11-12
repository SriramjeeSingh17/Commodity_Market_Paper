#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 16:02:16 2019

@author: ssingh17
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from pykalman import KalmanFilter

'''Kalman filtering for an equation with intercept and slope'''
data_final = pd.read_csv('M:/Documents/Third Year Paper/Research work/data_withProd_final.csv')
Price_change = data_final.loc[:, 'Price_change'].values
a = data_final.loc[:, 'Unanticipated_prod'].values
a = a.reshape(len(a), 1)
b = np.concatenate((a, np.ones(a.shape)), axis=1)
unanticipated_prod_mat = b[:, np.newaxis, :]


trans_cov = 0.5*np.eye(2)

kf = KalmanFilter(n_dim_obs = 1, n_dim_state = 2, initial_state_mean = [-1.06, 0], initial_state_covariance = 0.5*np.ones((2,2)), transition_matrices = np.eye(2), observation_matrices = unanticipated_prod_mat, observation_covariance = 0.5, transition_covariance = trans_cov) 
#kf = KalmanFilter(n_dim_obs = 1, n_dim_state = 2, initial_state_mean = [-0.5, 0.5], initial_state_covariance = 0.1*np.eye(2), transition_matrices = np.eye(2), observation_matrices = unanticipated_prod_mat, observation_covariance = 1.0, transition_covariance = trans_cov) 
state_means, state_covs = kf.filter(Price_change)
slope = state_means[:, 0]
intercept = state_means[:, 1]


df_coef = pd.DataFrame({'Slope Coefficient': slope, 'Intercept Coefficient': intercept})
df_coef['Period'] = np.arange(len(df_coef)) + 1

z = np.polyfit(df_coef['Period'], df_coef['Slope Coefficient'], 2)
p = np.poly1d(z)

plt.plot(df_coef['Period'], df_coef['Slope Coefficient'], label = 'Case 2')
plt.plot(df_coef['Period'], p(df_coef['Period']))
plt.xlabel('Time Period (1992(0) - 2019(132))') 
plt.ylabel('Slope coefficient') 
plt.legend(loc='upper right') 
plt.show()
r2_score(df_coef['Slope Coefficient'], p(df_coef['Period']))


'''LR suggests that intercept is not significant (KF for an equation with no intercept)'''
data_final = pd.read_csv('M:/Documents/Third Year Paper/Research work/data_withProd_final.csv')
Price_change = data_final.loc[:, 'Price_change'].values
a = data_final.loc[:, 'Unanticipated_prod'].values
b = a.reshape(len(a), 1)
unanticipated_prod_mat = b[:, np.newaxis, :]

kf = KalmanFilter(n_dim_obs = 1, n_dim_state = 1, initial_state_mean = -1.06, initial_state_covariance = 1.0, transition_matrices = 1, observation_matrices = unanticipated_prod_mat, observation_covariance = 1.5, transition_covariance = 0.1) 

state_means, state_covs = kf.filter(Price_change)
slope = state_means[:, 0]

df_coef = pd.DataFrame({'Slope Coefficient': slope})
df_coef['Period'] = np.arange(len(df_coef)) + 1

z = np.polyfit(df_coef['Period'], df_coef['Slope Coefficient'], 2)
p = np.poly1d(z)

plt.plot(df_coef['Period'], df_coef['Slope Coefficient'], label = 'Case 3')
plt.plot(df_coef['Period'], p(df_coef['Period']))
plt.xlabel('Time Period (1992(0) - 2019(132))') 
plt.ylabel('Slope coefficient') 
plt.legend(loc='upper right') 
plt.show()
r2_score(df_coef['Slope Coefficient'], p(df_coef['Period']))


