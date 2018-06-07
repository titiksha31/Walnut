import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression

data=pd.read_csv('out.csv', delimiter=',',index_col = False)

dataset =data['gray']
target =data['blur']

dataset_train, dataset_test, target_train, target_test = train_test_split(dataset, target, test_size=0.20, random_state=42)

regr = LinearRegression(fit_intercept=True)
regr.fit(dataset_train,target_train)

print('Coefficients: \n', regr.coef_)
