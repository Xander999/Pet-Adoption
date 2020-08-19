#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 15 01:07:02 2020

@author: xander999
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#---- This function tokenizes different spaced 'color_type' in the dataset-- 
def xx(x):
    x=x.strip()
    return x.split(' ')

def cc(x):
    x=str(x)
    st=x.split(' ')
    return st[0]
# =============================================================================
# Fetching the Training Data
# =============================================================================

dataset=pd.read_csv('Dataset/train.csv')

data=dataset.loc[:,['pet_id','condition','color_type','length(m)',
                   'height(cm)', 'X1', 'X2', 'breed_category',
                   'pet_category']]

data['dif']=pd.Series(pd.to_datetime(dataset.listing_date, format='%Y-%m-%d')
                           -pd.to_datetime(dataset.issue_date, format='%Y-%m-%d')).apply(cc).array

data=data.set_index('pet_id')

# =============================================================================
# #---------Missing Value in Training Data------------
# =============================================================================
data=data.fillna(data.mean().iloc[0])
data['color_type']=data['color_type'].apply(xx)
X_train1=data.loc[:,['condition', 'color_type', 'X1', 'X2', 'dif']]
Y_train1=data.loc[:,['breed_category']]  
Y_train2=data.loc[:,['pet_category']]
# =============================================================================
# #---------Categorical Values---------
# =============================================================================
X_train1=pd.get_dummies(X_train1, prefix=['Condition_'], columns=['condition'])
aa=pd.get_dummies(X_train1['color_type'].apply(pd.Series).stack(), 
                        columns=['color_type']).sum(level=0)

X_train1=pd.concat([X_train1, aa], axis=1, sort=False)
X_train1 = X_train1.drop(['color_type'], axis=1)

from keras.utils import to_categorical
Y_train1 = to_categorical(Y_train1)
Y_train2 = to_categorical(Y_train2)
Y_train2 = np.delete(Y_train2, 3, axis=1)
# =============================================================================
# ---------------------Fetching the Testing Data
# =============================================================================

dataset1=pd.read_csv('Dataset/test.csv')

data1=dataset1.loc[:,['pet_id','condition','color_type','length(m)',
                   'height(cm)', 'X1', 'X2']]

data1['dif']=pd.Series(pd.to_datetime(dataset1.listing_date, format='%Y-%m-%d')
                           -pd.to_datetime(dataset1.issue_date, format='%Y-%m-%d')).apply(cc).array

data1=data1.set_index('pet_id')
sm=dataset1.loc[:,'pet_id']
sm=pd.DataFrame(sm)
# =============================================================================
# #---------Missing Value in Testing Data------------
# =============================================================================
data1=data1.fillna(data1.mean().iloc[0])
data1['color_type']=data1['color_type'].apply(xx)
X_test1=data1.loc[:,['condition', 'color_type', 'X1', 'X2', 'dif']]
# =============================================================================
# #---------Categorical Values in Test Data---------
# =============================================================================
X_test1=pd.get_dummies(X_test1, prefix=['Condition_'], columns=['condition'])
aa=pd.get_dummies(X_test1['color_type'].apply(pd.Series).stack(), 
                        columns=['color_type']).sum(level=0)

X_test1=pd.concat([X_test1, aa], axis=1, sort=False)
X_test1 = X_test1.drop(['color_type'], axis=1)

# =============================================================================
# #---------Feature Scaling
# =============================================================================
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train1 = sc_X.fit_transform(X_train1)
X_test1=sc_X.transform(X_test1)




import keras
from keras.models import Sequential
from keras.layers import Dense

# =============================================================================
# Breed Claassification
# =============================================================================

# Initialising the ANN
classifier1 = Sequential()

# Adding the input layer and the first hidden layer
classifier1.add(Dense(units = 43, kernel_initializer  = 'uniform', activation = 'relu', input_shape = (43,)))

# Adding the second hidden layer
classifier1.add(Dense(units = 43, kernel_initializer = 'uniform', activation = 'relu'))

# Adding the third hidden layer
classifier1.add(Dense(units = 43, kernel_initializer = 'uniform', activation = 'relu'))

# Adding the output layer
classifier1.add(Dense(units = 3, kernel_initializer = 'uniform', activation = 'softmax'))

# Compiling the ANN
classifier1.compile(optimizer = 'sgd', loss = 'categorical_crossentropy', metrics = ['accuracy'])
# optimizer= rmsprop/adam/
# Fitting the ANN to the Training set

classifier1.fit(X_train1, Y_train1, batch_size = 110, epochs = 160)
# =============================================================================
# classifier.fit(X_train, y_train, batch_size = 15, nb_epoch = 60)
# classifier.fit(X_train, y_train, batch_size = 25, nb_epoch = 90)
# =============================================================================

# Predicting the Test set results
y_pred = classifier1.predict(X_test1)
y_pred = pd.DataFrame(y_pred)
y_pred[0] = y_pred[0].apply(lambda x: 1 if (x > 0.5) else 0)
y_pred[1] = y_pred[1].apply(lambda x: 1 if (x > 0.5) else 0)
y_pred[2] = y_pred[2].apply(lambda x: 1 if (x > 0.5) else 0)

z=y_pred[y_pred==1].stack().reset_index().drop(0,1)
z=z.drop('level_0', axis=1)
z=pd.concat([z, sm], axis=1, sort=False)
z=z.set_index('pet_id')
z=z.rename(columns={'level_1':'breed_category'})

# =============================================================================
# Pet Claassification
# =============================================================================

# Initialising the ANN
classifier2 = Sequential()

# Adding the input layer and the first hidden layer
classifier2.add(Dense(units = 43, kernel_initializer  = 'uniform', activation = 'relu', input_shape = (43,)))

# Adding the second hidden layer
classifier2.add(Dense(units = 43, kernel_initializer = 'uniform', activation = 'relu'))

# Adding the third hidden layer
classifier2.add(Dense(units = 43, kernel_initializer = 'uniform', activation = 'relu'))

# Adding the output layer
classifier2.add(Dense(units = 4, kernel_initializer = 'uniform', activation = 'softmax'))

# Compiling the ANN
classifier2.compile(optimizer = 'sgd', loss = 'categorical_crossentropy', metrics = ['accuracy'])
# optimizer= rmsprop/adam/
# Fitting the ANN to the Training set

classifier2.fit(X_train1, Y_train2, batch_size = 110, epochs = 160)
# =============================================================================
# classifier.fit(X_train, y_train, batch_size = 15, nb_epoch = 60)
# classifier.fit(X_train, y_train, batch_size = 25, nb_epoch = 90)
# =============================================================================