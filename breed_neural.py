#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 14 21:36:11 2020

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

# =============================================================================
# #---------Splitting the dataset into the Training set and Test set
# =============================================================================
from sklearn.model_selection import train_test_split
X_train11, X_test11, Y_train11, Y_test11 = train_test_split(X_train1, Y_train1, test_size = 0.2, random_state = 0)

# =============================================================================
# #---------Feature Scaling
# =============================================================================
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train11 = sc_X.fit_transform(X_train11)
X_test11 = sc_X.transform(X_test11)


import keras
from keras.models import Sequential
from keras.layers import Dense

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(units = 43, kernel_initializer  = 'uniform', activation = 'relu', input_shape = (43,)))

# Adding the second hidden layer
classifier.add(Dense(units = 43, kernel_initializer = 'uniform', activation = 'relu'))

# Adding the third hidden layer
classifier.add(Dense(units = 43, kernel_initializer = 'uniform', activation = 'relu'))

# Adding the output layer
classifier.add(Dense(units = 3, kernel_initializer = 'uniform', activation = 'softmax'))

# Compiling the ANN
classifier.compile(optimizer = 'sgd', loss = 'categorical_crossentropy', metrics = ['accuracy'])
# optimizer= rmsprop/adam/
# Fitting the ANN to the Training set

classifier.fit(X_train11, Y_train11, batch_size = 110, epochs = 160)
# =============================================================================
# classifier.fit(X_train, y_train, batch_size = 15, nb_epoch = 60)
# classifier.fit(X_train, y_train, batch_size = 25, nb_epoch = 90)
# =============================================================================



# Predicting the Test set results
y_pred = classifier.predict(X_test11)
y_pred = pd.DataFrame(y_pred)
y_pred[0] = y_pred[0].apply(lambda x: 1 if (x > 0.5) else 0)
y_pred[1] = y_pred[1].apply(lambda x: 1 if (x > 0.5) else 0)
y_pred[2] = y_pred[2].apply(lambda x: 1 if (x > 0.5) else 0)


Y_test11=pd.DataFrame(Y_test11)
z=y_pred[y_pred==1].stack().reset_index().drop(0,1)
z1=Y_test11[Y_test11==1].stack().reset_index().drop(0,1)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test11, y_pred)

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(Y_test11, y_pred)
print(accuracy*100)
