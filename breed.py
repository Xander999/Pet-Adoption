#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  9 22:06:16 2020

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


# =============================================================================
# #-------------------Fitting K-NN to the Training set---------------------
# =============================================================================
from sklearn.neighbors import KNeighborsClassifier
knn_cl_breed = KNeighborsClassifier(n_neighbors = 3, metric = 'minkowski', p = 2)


#------For X_train11 and Y_train11-------------
knn_cl_breed.fit(X_train11, Y_train11['breed_category'])

# Predicting the Test set results
Y_pred11_breed_category = knn_cl_breed.predict(X_test11)

# Making the Confusion Matrix and Accuracy Score
from sklearn.metrics import confusion_matrix, accuracy_score
knn_cm_breed11 = confusion_matrix(Y_test11['breed_category'], Y_pred11_breed_category)
knn_acc_breed11=accuracy_score(Y_test11['breed_category'], Y_pred11_breed_category)




# =============================================================================
# #-------------------Fitting Logistic Regression to the Training set----------
# =============================================================================
from sklearn.linear_model import LogisticRegression
logReg_cl_breed = LogisticRegression(random_state = 0, max_iter=7000)


#------For X_train11 and Y_train11-------------
logReg_cl_breed.fit(X_train11, Y_train11['breed_category'])

# Predicting the Test set results
Y_pred11_breed_category = logReg_cl_breed.predict(X_test11)

# Making the Confusion Matrix and Accuracy Score
from sklearn.metrics import confusion_matrix, accuracy_score
log_cm_breed11 = confusion_matrix(Y_test11['breed_category'], Y_pred11_breed_category)
log_acc_breed11=accuracy_score(Y_test11['breed_category'], Y_pred11_breed_category)



# =============================================================================
# #-----------------------Fitting SVM to the Training set---------------------
# =============================================================================
from sklearn.svm import SVC
SVC_cl_breed = SVC(kernel = 'linear', random_state = 0)


#------For X_train11 and Y_train11-------------
SVC_cl_breed.fit(X_train11, Y_train11['breed_category'])

# Predicting the Test set results
Y_pred11_breed_category = SVC_cl_breed.predict(X_test11)

# Making the Confusion Matrix and Accuracy Score
from sklearn.metrics import confusion_matrix, accuracy_score
SVC_cm_breed11 = confusion_matrix(Y_test11['breed_category'], Y_pred11_breed_category)
SVC_acc_breed11=accuracy_score(Y_test11['breed_category'], Y_pred11_breed_category)



# =============================================================================
# # ----------------Fitting Naive Bayes to the Training set-----------------------
# =============================================================================
from sklearn.naive_bayes import GaussianNB
Gaus_cl_breed =  GaussianNB()


#------For X_train11 and Y_train11-------------
Gaus_cl_breed.fit(X_train11, Y_train11['breed_category'])

# Predicting the Test set results
Y_pred11_breed_category = Gaus_cl_breed.predict(X_test11)

# Making the Confusion Matrix and Accuracy Score
from sklearn.metrics import confusion_matrix, accuracy_score
Gaus_cm_breed11 = confusion_matrix(Y_test11['breed_category'], Y_pred11_breed_category)
Gaus_acc_breed11=accuracy_score(Y_test11['breed_category'], Y_pred11_breed_category)



# =============================================================================
# #---------------Fitting Decision Tree Classification to the Training set------
# =============================================================================
from sklearn.tree import DecisionTreeClassifier
DTC_cl_breed =  DecisionTreeClassifier(criterion = 'gini', random_state = 0)


#------For X_train11 and Y_train11-------------
DTC_cl_breed.fit(X_train11, Y_train11['breed_category'])

# Predicting the Test set results
Y_pred11_breed_category = DTC_cl_breed.predict(X_test11)

# Making the Confusion Matrix and Accuracy Score
from sklearn.metrics import confusion_matrix, accuracy_score
DTC_cm_breed11 = confusion_matrix(Y_test11['breed_category'], Y_pred11_breed_category)
DTC_acc_breed11=accuracy_score(Y_test11['breed_category'], Y_pred11_breed_category)


# =============================================================================
# After Evaluation of 5 different Models we have accuracy for each models  which 
# we will make on the DataFrame nameing "Accurracy_model". And then we can display
# it on the graph.
# =============================================================================



breed_acc=[knn_acc_breed11*100, log_acc_breed11*100, SVC_acc_breed11*100, Gaus_acc_breed11*100, DTC_acc_breed11*100]
rows=['KNN', 'Logistic Regression', 'SVC', 'Naive Bayes', 'DecisonTreeClassification']

Accuracy_model5=pd.DataFrame(data={'Breed':breed_acc}, index=rows)

ax=Accuracy_model5.plot.bar(figsize=(12,9), rot=0, bottom=81)


# =============================================================================
# Accuracy Model1 : 
# condition','color_type','length(m)', 'height(cm)', 'X1', 'X2', 'dif'
# =============================================================================

# =============================================================================
# Accuracy Model2:
# 'condition','color_type','length(m)', 'height(cm)', 'dif'
# =============================================================================

# =============================================================================
# Accuracy Model3 :
# 'condition','color_type','length(m)', 'height(cm)', 'X1', 'dif'
# =============================================================================

# =============================================================================
# Accuracy Model4:
# 'condition', 'color_type', 'length(m)', 'X1', 'X2', 'dif'
# =============================================================================

# =============================================================================
# Accuracy Model5:  Best Fit  (90.54)
# 'condition', 'color_type', 'X1', 'X2', 'dif'
# =============================================================================

