# -*- coding: utf-8 -*-
"""
Created on Mon Jul 16 20:50:41 2018

@author: Kevin D'Cruz
"""

# Data Preprocessing

#Importing the Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#Importing the Dataset
dataset=pd.read_csv('Data.csv')
#Creating Matrix of Features(Independent Variables)
X= dataset.iloc[:, :-1].values #-1 means exluding the last column
Y= dataset.iloc[:, 3].values #3 is index of last column (dependent variable)

"""
#Taking care of Missing Data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values= 'NaN', strategy = 'mean', axis=0)
imputer = imputer.fit(X[:, 1:3]) #Index 1 and 2, fit imputer object to matrix X
X[:, 1:3]=imputer.transform(X[:, 1:3])#Replaces missing data with mean obtained above



#Encoding Categorical Data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0]) #fitted object labelencoder_X and encoded it
onehotencoder = OneHotEncoder(categorical_features=[0])
X = onehotencoder.fit_transform(X).toarray()

labelencoder_Y = LabelEncoder() #OneHotEncoder not required coz dependent variable
Y = labelencoder_Y.fit_transform(Y) #fitted object labelencoder_y and encoded it
"""

#Split to training and test set
from sklearn.cross_validation import train_test_split #canbe replaced by sklearn.model_selection
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0) 

"""
#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test) #fitting not required, coz already fitted with training set
"""