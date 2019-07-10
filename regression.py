#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@author: Niwedita J
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Importing the dataset
dataset = pd.read_csv('sub_item.csv')

# Changing string value to integer
dataset["inner_height_mm"] = map(lambda x: float(x.replace("\"", "" )), dataset["inner_height_mm"])
dataset["height_inch"] = map(lambda x: float(x.replace("\"", "" )), dataset["height_inch"])
dataset["inner_height"] = map(lambda x: float(x.replace("\"", "" )), dataset["inner_height"])
dataset["length_inch"] = map(lambda x: float(x.replace("\"", "" )), dataset["length_inch"])

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder 
labelencoder_dataset = LabelEncoder()
dataset["sku_code"] = labelencoder_dataset.fit_transform(dataset["sku_code"])
dataset["material"] = labelencoder_dataset.fit_transform(dataset["material"])
dataset["box_code"] = labelencoder_dataset.fit_transform(dataset["box_code"])
dataset["bursting_factor"] = labelencoder_dataset.fit_transform(dataset["bursting_factor"])
dataset["flute_type"] = labelencoder_dataset.fit_transform(dataset["flute_type"])
dataset["updated_at"] = labelencoder_dataset.fit_transform(dataset["updated_at"])
dataset["company_id"] = labelencoder_dataset.fit_transform(dataset["company_id"])


X = dataset.iloc[:, :-1].values #X is the model_data
y = dataset.iloc[:, 13].values #y is the target data


# Taking care of missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X[:, 0:12])
X[:, 0:12] = imputer.transform(X[:, 0:12])

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)


# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor_l = LinearRegression()
regressor_l.fit(X_train, y_train)

# Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg_2 = PolynomialFeatures(degree = 2)
X_poly_2 = poly_reg_2.fit_transform(X_train)
poly_reg_2.fit(X_poly_2, y_train)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly_2, y_train)

# Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg_3 = PolynomialFeatures(degree = 3)
X_poly_3 = poly_reg_3.fit_transform(X_train)
poly_reg_3.fit(X_poly_3, y_train)
lin_reg_3 = LinearRegression()
lin_reg_3.fit(X_poly_3, y_train)

# Fitting Polynomial Regression to the dataset
'''from sklearn.preprocessing import PolynomialFeatures
poly_reg_4 = PolynomialFeatures(degree = 4)
X_poly_4 = poly_reg_4.fit_transform(X_train)
poly_reg_4.fit(X_poly_4, y_train)
lin_reg_4 = LinearRegression()
lin_reg_4.fit(X_poly_4, y_train)'''

# Fitting SVR to the dataset
from sklearn.svm import SVR
regressor_svr = SVR(kernel = 'rbf')
regressor_svr.fit(X_train, y_train)

# Fitting Decision Tree Regression to the dataset
from sklearn.tree import DecisionTreeRegressor
regressor_dtr = DecisionTreeRegressor(random_state = 0)
regressor_dtr.fit(X_train, y_train)


# Predicting the Test set results
y_pred_l = regressor_l.predict(X_test) #Prediction using linear regresssion
y_pred_p2=lin_reg_2.predict(poly_reg_2.fit_transform(X_test)) #Prediction using polynomial regression(degree=2)
y_pred_p3=lin_reg_3.predict(poly_reg_3.fit_transform(X_test)) #Prediction using polynomial regression(degree=3)
#y_pred_p4=lin_reg_4.predict(poly_reg_4.fit_transform(X_test)) ##Prediction using polynomial regression(degree=4)
y_pred_svr = regressor_svr.predict(X_test) #Prediction using support vector regression
y_pred = regressor_dtr.predict(X_test) #Prediction using decision tree regression

'''# Predicting result giving a different input
v = pd.DataFrame([-0.31960567,  0.48685133,  0.18608074,  0.48864152, -0.54070714,
       -0.21581051, -0.1728518 ,  0.24106584,  0.2686961 ,  0.49009078,
       -0.52680201, -0.36838354,  1.04291219])
y_pred_r = regressor_l.predict(v[0].values.reshape(1,-1))'''
