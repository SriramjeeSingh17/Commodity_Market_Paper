# -*- coding: utf-8 -*-
"""
Created on Sat Jul 30 19:44:48 2022

@author: ssingh17
"""

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor 
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')


'''Regression Model'''
data_final = pd.read_csv('M:/Documents/Third Year Paper/Research work/data_withProd_final.csv')
data_final.count()
#data_final = data_final[data_final['Year'] >= 2000]
data_final = data_final[data_final['Months'] == 8]
data_final_train = data_final[(data_final['Year'] < 2016)]
data_final_test = data_final[(data_final['Year'] >= 2016)]
df_train_new = data_final_train[['Date', 'Unanticipated_prod', 'Price_change']]
df_test_new = data_final_test[['Date', 'Unanticipated_prod', 'Price_change']]
X_train, X_test = df_train_new.iloc[:, 1:2].values, df_test_new.iloc[:, 1:2].values
y_train, y_test = df_train_new.iloc[:, -1:].values, df_test_new.iloc[:, -1:].values

sc = MinMaxScaler(feature_range=(0, 1))
X_train, y_train = sc.fit_transform(X_train), sc.fit_transform(y_train)
X_test = sc.transform(X_test)

#df_new = data_final[['Date', 'Unanticipated_prod', 'Price_change']]
#X = df_new.iloc[:, 1:2].values
#y = df_new.iloc[:, -1:].values
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)

model = LinearRegression()
model.fit(X_train,y_train)
print(model.intercept_)
print(model.coef_)
y_pred=model.predict(X_test)

y_pred = sc.inverse_transform(y_pred)
mse = mean_squared_error(y_test,y_pred)
mae = mean_absolute_error(y_test,y_pred)
print('MSE: %.2f' % mse, 'MAE: %.2f' % mae)


'''Support Vector Regression'''
model = SVR(kernel='linear')
model.fit(X_train, y_train.ravel())
y_pred = model.predict(X_test)
y_pred = y_pred.reshape((len(y_pred), 1))

y_pred = sc.inverse_transform(y_pred)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test,y_pred)
print('MSE: %.2f' % mse, 'MAE: %.2f' % mae)


'''Random Forest Regression''' 
model = RandomForestRegressor(n_estimators = 100) 
model.fit(X_train, y_train.ravel())
y_pred = model.predict(X_test)
y_pred = y_pred.reshape((len(y_pred), 1))

y_pred = sc.inverse_transform(y_pred)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test,y_pred)
print('MSE: %.2f' % mse, 'MAE: %.2f' % mae)


'''Locally Weighted Linear Regression'''
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
    X_ = np.append(X, np.ones(m).reshape(m,1), axis=1) 
    point_ = np.array([point, 1]) 
    w = wm(point_, X_, tau) 
    theta = np.linalg.pinv(X_.T*(w * X_))*(X_.T*(w * y))   
    pred = np.dot(point_, theta) 
    return theta, pred

y_pred = []
for i in range(len(X_test)):
    coefficients, preds = predict(X_train, y_train, X_test[i][0], 5)
    pred_value = preds[0, 0]
    pred_value = np.array(pred_value)
    y_pred.append(pred_value)

y_pred = sc.inverse_transform(y_pred)    
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test,y_pred)
print('MSE: %.2f' % mse, 'MAE: %.2f' % mae)


'''Fully Connected Neural Networks'''
inputs = keras.Input(shape = (1, ), name = 'my_input')
features = layers.Dense(64, activation = 'linear')(inputs)
features = layers.Dropout(0.5)(features)
#features = layers.Dense(32, activation = 'linear')(features)
outputs = layers.Dense(1, activation = 'linear')(features)
model = keras.Model(inputs = inputs, outputs = outputs)

model.summary()

model.compile(optimizer = 'adam', loss = 'mean_squared_error')
hist = model.fit(X_train, y_train, batch_size = 16, epochs = 1000, verbose = 0, validation_data = (X_test, y_test))

plt.plot(hist.history['loss'], 'r--')
plt.plot(hist.history['val_loss'], 'b-')
plt.title('Model Losses')
plt.ylabel('Losses')
plt.xlabel('Epochs')
plt.legend(['Train', 'Test'], loc='upper right')
plt.show()

y_pred = model.predict(X_test)
y_pred = sc.inverse_transform(y_pred)
mse = mean_squared_error(y_test,y_pred)
mae = mean_absolute_error(y_test,y_pred)
print('MSE: %.2f' % mse, 'MAE: %.2f' % mae)

