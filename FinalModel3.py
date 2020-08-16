#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  9 22:06:16 2020

@author: xander999
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



# =============================================================================
# Fetching the Trraining Data
# =============================================================================

dataset=pd.read_csv('Dataset/train.csv')

data=dataset.loc[:,['pet_id','condition','color_type','length(m)',
                   'height(cm)', 'X1', 'X2', 'breed_category',
                   'pet_category']]
data=data.set_index('pet_id')

# =============================================================================
# #---------Missing Value in Training Data------------
# =============================================================================
data1=data[~data['condition'].isna()]
X_train1=data1.loc[:,['condition','color_type','length(m)',
                   'height(cm)', 'X1', 'X2']]
Y_train1=data1.loc[:,['breed_category', 'pet_category']]
# =============================================================================
# #---------Categorical Values in Train Data---------
# =============================================================================
X_train1=pd.get_dummies(X_train1, prefix=['Condition_'], columns=['condition'])
X_train1=pd.get_dummies(X_train1, prefix=['Color_'], columns=['color_type'])



# =============================================================================
# ---------------------Fetching the Testing Data
# =============================================================================

dataset1=pd.read_csv('Dataset/test.csv')

data1=dataset1.loc[:,['pet_id','condition','color_type','length(m)',
                   'height(cm)', 'X1', 'X2']]

data1=data1.set_index('pet_id')
sm=dataset1.loc[:,'pet_id']

sm=pd.DataFrame(sm)
# =============================================================================
# #---------Categorical Values in Test Data---------
# =============================================================================
X_test1=data1.iloc[:,:]
X_test1=pd.get_dummies(X_test1, prefix=['Condition_'], columns=['condition'])
X_test1=pd.get_dummies(X_test1, prefix=['Color_'], columns=['color_type'])



# =============================================================================
# The color_type for train_x and pred_x have different colorr combination.
# In order to predict it we need a common color_type dummy matrix for both 
# training and predicting independent variables.
# =============================================================================

l1=X_train1.iloc[:,7:].columns
l2=X_test1.iloc[:,7:].columns

a1=[]
b1=[]

for x in l2:
    if x not in l1:
        a1.append(x)
for x in l1:
    if x not in l2:
        
        b1.append(x)
        
# =============================================================================
# Now a1 consist of columns that are in pred_x but not in train1_x. So we add  
# the particular column in train1_x. Similarly the columns in b1 are added in 
# dataframe pred_x.
# =============================================================================

for x in a1:
    X_train1[x]=[0]*len(X_train1.index)
for x in b1:
    X_test1[x]=[0]*len(X_test1.index)
                     




# =============================================================================
# #---------Feature Scaling in Training Data-----------------
# =============================================================================
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train1 = sc_X.fit_transform(X_train1)

# =============================================================================
# #---------Feature Scaling in Testing Data-----------------
# =============================================================================
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_test1 = sc_X.fit_transform(X_test1)

# =============================================================================
# ---------------Fitting Decision Tree Classification to the Training set
#---------------------------For Breed Classification-------------
# =============================================================================
from sklearn.tree import DecisionTreeClassifier
DTC_cl_breed =  DecisionTreeClassifier(criterion = 'entropy', random_state = 0)

#------For X_train11 and Y_train11-------------
DTC_cl_breed.fit(X_train1, Y_train1['breed_category'])
# =============================================================================
# #-----------------------Fitting SVM to the Training set---------------------
#---------------------------For Pet Classification----------------------------- 
# =============================================================================
from sklearn.svm import SVC
SVC_cl_pet = SVC(kernel = 'linear', random_state = 0)

#------For X_train11 and Y_train11-------------
SVC_cl_pet.fit(X_train1, Y_train1['pet_category'])




# =============================================================================
# Prediction of breed category and Pet category
# =============================================================================

Y_pred1_breed_category = DTC_cl_breed.predict(X_test1)
Y_pred1_pet_category = SVC_cl_pet.predict(X_test1)

Y_pred1_breed_category = Y_pred1_breed_category.astype(np.int64)
brd=pd.DataFrame(Y_pred1_breed_category)

dff=pd.DataFrame(data={'breed_category':Y_pred1_breed_category, 
                       'pet_category': Y_pred1_pet_category})

dff= pd.concat([dff, sm], axis=1, sort=False)
dff=dff.set_index('pet_id')
dff.to_csv('Result3.csv')
