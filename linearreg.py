import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
data=pd.read_csv('out.csv', delimiter=',',index_col = False)
data=data[40000:40600]
#print (data['red'])
#print(len(data))
#data_r=data.red
data_r=data[['red']]
data_gr=data[['green']]
data_b=data[['blue']]
#data_r=data_r.values
data_r_train= data_r[:200]
data_r_test = data_r[200:]

data_gr_train= data_gr[:200]
data_gr_test = data_gr[200:]

data_b_train= data_b[:200]
data_b_test = data_b[200:]
#print(len(data_X_test))
#print (data_X_train)
#data_g=data.gray
data_g=data[['gray']]
#print (data_g)
data_y_train = data_g[:200]
data_y_test = data_g[200:]
#data_X_train.reshape((-1,1))
#data_y_train.reshape((1,-1))
regr_r = linear_model.LinearRegression()
regr_r.fit(data_r_train, data_y_train)
data_y_pred_r = regr_r.predict(data_r_test)
print('Coefficients: \n', regr_r.coef_)
print("Mean squared error: %.2f"%mean_squared_error(data_y_test, data_y_pred_r))
print('Variance score: %.2f' % r2_score(data_y_test, data_y_pred_r))

regr_gr = linear_model.LinearRegression()
regr_gr.fit(data_gr_train, data_y_train)
data_y_pred_gr = regr_gr.predict(data_gr_test)
print('Coefficients: \n', regr_gr.coef_)
print("Mean squared error: %.2f"%mean_squared_error(data_y_test, data_y_pred_gr))
print('Variance score: %.2f' % r2_score(data_y_test, data_y_pred_gr))

regr_b = linear_model.LinearRegression()
regr_b.fit(data_b_train, data_y_train)
data_y_pred_b = regr_b.predict(data_b_test)
print('Coefficients: \n', regr_b.coef_)
print("Mean squared error: %.2f"%mean_squared_error(data_y_test, data_y_pred_b))
print('Variance score: %.2f' % r2_score(data_y_test, data_y_pred_b))

#coef =[regr_r.coef_,regr_gr.coef_,regr_b.coef_]
#print(coef)
