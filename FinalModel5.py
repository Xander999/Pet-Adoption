#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 00:04:14 2020

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
Y_train1=data.loc[:,['breed_category', 'pet_category']]

# =============================================================================
# #---------Categorical Values---------
# =============================================================================
X_train1=pd.get_dummies(X_train1, prefix=['Condition_'], columns=['condition'])
aa=pd.get_dummies(X_train1['color_type'].apply(pd.Series).stack(), 
                        columns=['color_type']).sum(level=0)

X_train1=pd.concat([X_train1, aa], axis=1, sort=False)
X_train1 = X_train1.drop(['color_type'], axis=1)

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



# =============================================================================
# #---------------Fitting Decision Tree Classification to the Training set-----
#------------------------------Breed Classification--------------------------
# =============================================================================
from sklearn.tree import DecisionTreeClassifier
DTC_cl_breed =  DecisionTreeClassifier(criterion = 'gini', random_state = 0)


#------For X_train11 and Y_train11-------------
DTC_cl_breed.fit(X_train1, Y_train1['breed_category'])


# =============================================================================
# #-------------------Fitting K-NN to the Training set---------------------
#---------------------------Pet Classification---------------------
# =============================================================================
from sklearn.neighbors import KNeighborsClassifier
knn_cl_pet = KNeighborsClassifier(n_neighbors =5, metric = 'minkowski', p = 2)

#------For X_train11 and Y_train11-------------
knn_cl_pet.fit(X_train1, Y_train1['pet_category'])



# =============================================================================
# Prediction of breed category and Pet category
# =============================================================================

Y_pred1_breed_category = DTC_cl_breed.predict(X_test1)
Y_pred1_pet_category = knn_cl_pet.predict(X_test1)

Y_pred1_breed_category = Y_pred1_breed_category.astype(np.int64)
brd=pd.DataFrame(Y_pred1_breed_category)

dff=pd.DataFrame(data={'breed_category':Y_pred1_breed_category, 
                       'pet_category': Y_pred1_pet_category})

dff= pd.concat([dff, sm], axis=1, sort=False)
dff=dff.set_index('pet_id')
dff.to_csv('Result5.csv')





