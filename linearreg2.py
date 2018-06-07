import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import linear_model, model_selection
from sklearn.metrics import mean_squared_error, r2_score
data=pd.read_csv('out.csv', delimiter=',',index_col = False)

data2=data[['blue','green','red']]
data1=data['gray']

X_train, X_test, y_train, y_test = model_selection.train_test_split(data2, data1, test_size=0.20)


regr = linear_model.LinearRegression()
regr.fit(X_train, y_train)
data_y_pred_r = regr.predict(X_test)
print('Coefficients: \n', regr.coef_)



