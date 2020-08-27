#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 22:59:27 2020

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
#data=data.fillna(data.mean().iloc[0])
data=data.fillna(0)
data['color_type']=data['color_type'].apply(xx)
X_train1=data.loc[:,['condition','color_type','X1', 'X2', 'dif']]
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
knn_cl_pet = KNeighborsClassifier(n_neighbors =5, metric = 'minkowski', p = 2)


#------For X_train11 and Y_train11-------------
knn_cl_pet.fit(X_train11, Y_train11['pet_category'])

# Predicting the Test set results
Y_pred11_pet_category = knn_cl_pet.predict(X_test11)

# Making the Confusion Matrix and Accuracy Score
from sklearn.metrics import confusion_matrix, accuracy_score

knn_cm_pet11 = confusion_matrix(Y_test11['pet_category'], Y_pred11_pet_category)
knn_acc_pet11=accuracy_score(Y_test11['pet_category'], Y_pred11_pet_category)






# =============================================================================
# #-------------------Fitting Logistic Regression to the Training set----------
# =============================================================================
from sklearn.linear_model import LogisticRegression
logReg_cl_pet = LogisticRegression(random_state=0, max_iter=7000)


#------For X_train11 and Y_train11-------------
logReg_cl_pet.fit(X_train11, Y_train11['pet_category'])

# Predicting the Test set results
Y_pred11_pet_category = logReg_cl_pet.predict(X_test11)

# Making the Confusion Matrix and Accuracy Score
from sklearn.metrics import confusion_matrix, accuracy_score
log_cm_pet11 = confusion_matrix(Y_test11['pet_category'], Y_pred11_pet_category)
log_acc_pet11=accuracy_score(Y_test11['pet_category'], Y_pred11_pet_category)








# =============================================================================
# #-----------------------Fitting SVM to the Training set---------------------
# =============================================================================
from sklearn.svm import SVC
SVC_cl_pet = SVC(kernel = 'rbf', random_state = 0)


#------For X_train11 and Y_train11-------------
SVC_cl_pet.fit(X_train11, Y_train11['pet_category'])

# Predicting the Test set results
Y_pred11_pet_category = SVC_cl_pet.predict(X_test11)

# Making the Confusion Matrix and Accuracy Score
from sklearn.metrics import confusion_matrix, accuracy_score
SVC_cm_pet11 = confusion_matrix(Y_test11['pet_category'], Y_pred11_pet_category)
SVC_acc_pet11=accuracy_score(Y_test11['pet_category'], Y_pred11_pet_category)








# =============================================================================
# # ----------------Fitting Naive Bayes to the Training set-----------------------
# =============================================================================
from sklearn.naive_bayes import GaussianNB
Gaus_cl_pet =  GaussianNB()


#------For X_train11 and Y_train11-------------
Gaus_cl_pet.fit(X_train11, Y_train11['pet_category'])

# Predicting the Test set results
Y_pred11_pet_category = Gaus_cl_pet.predict(X_test11)

# Making the Confusion Matrix and Accuracy Score
from sklearn.metrics import confusion_matrix, accuracy_score
Gaus_cm_pet11 = confusion_matrix(Y_test11['pet_category'], Y_pred11_pet_category)
Gaus_acc_pet11=accuracy_score(Y_test11['pet_category'], Y_pred11_pet_category)






# =============================================================================
# #---------------Fitting Decision Tree Classification to the Training set------
# =============================================================================
from sklearn.tree import DecisionTreeClassifier
DTC_cl_pet =  DecisionTreeClassifier(criterion = 'entropy', random_state = 0)


#------For X_train11 and Y_train11-------------
DTC_cl_pet.fit(X_train11, Y_train11['pet_category'])

# Predicting the Test set results
Y_pred11_pet_category = DTC_cl_pet.predict(X_test11)

# Making the Confusion Matrix and Accuracy Score
from sklearn.metrics import confusion_matrix, accuracy_score
DTC_cm_pet11 = confusion_matrix(Y_test11['pet_category'], Y_pred11_pet_category)
DTC_acc_pet11=accuracy_score(Y_test11['pet_category'], Y_pred11_pet_category)



# =============================================================================
# Here we will be fitting Extreme Boost Algorithm
# =============================================================================
import xgboost as xgb

xg_reg = xgb.XGBClassifier(objective ='reg:logistic', 
                           colsample_bytree = 0.3, 
                           learning_rate = 1.5,
                           max_depth = 5, 
                           alpha = 14, 
                           n_estimators = 10)

# =============================================================================
# The present combination increases the accuracy of the model by 89.67
# =============================================================================
# objective ='reg:logistic', 
# colsample_bytree = 0.3, 
# learning_rate = 1.5,
# max_depth = 5, 
# alpha = 14, 
# n_estimators = 10
# =============================================================================
# =============================================================================
# Predicting the Test set results
xg_reg.fit(X_train11, Y_train11['pet_category'])
Y_pred11_pet_category = xg_reg.predict(X_test11)

# Making the Confusion Matrix and Accuracy Score
from sklearn.metrics import confusion_matrix, accuracy_score
XBOOST_cm_pet11 = confusion_matrix(Y_test11['pet_category'], Y_pred11_pet_category)
XBOOST_acc_pet11=accuracy_score(Y_test11['pet_category'], Y_pred11_pet_category)




# =============================================================================
# After Evaluation of 5 different Models we have accuracy for each models  which 
# we will make on the DataFrame nameing "Accurracy_model". And then we can display
# it on the graph.
# =============================================================================



pet_acc= [knn_acc_pet11*100, log_acc_pet11*100, SVC_acc_pet11*100, Gaus_acc_pet11*100, DTC_acc_pet11*100, XBOOST_acc_pet11*100]
rows=['KNN', 'Logistic Regression', 'SVC', 'Naive Bayes', 'DecisonTreeClassification','Xgboost']

Accuracy_model3=pd.DataFrame(data={'Pet': pet_acc}, index=rows)

ax=Accuracy_model3.plot.bar(figsize=(12,9), rot=0, bottom=81)


# =============================================================================
# Accuracy Model 1:
# 'condition','color_type','length(m)','height(cm)', 'X1', 'X2', 'dif'
# =============================================================================
# =============================================================================
# KNN	84.97478099283249
# Logistic Regression	86.43482877621449
# SVC	86.03663392620122
# Naive Bayes	34.61640562782055
# DecisonTreeClassification	85.21369790284045
#
# Logistic Regression	87.92142288293071
# SVC	87.84178391292807
# DecisonTreeClassification	85.87735598619591
# KNN	85.31988319617733
# Naive Bayes	36.740111494558
# =============================================================================


# =============================================================================
# Accuracy Model 2:
# 'condition','color_type','height(cm)', 'X1', 'X2', 'dif'
# =============================================================================
# =============================================================================
# Logistic Regression	87.92142288293071
# SVC	87.86833023626228
# KNN	87.41704273958057
# DecisonTreeClassification	86.03663392620122
# Naive Bayes	36.740111494558
# 
# =============================================================================
# =============================================================================
# KNN	87.41704273958057
# Logistic Regression	87.94796920626493
# SVC	87.86833023626228
# Naive Bayes	36.740111494558
# DecisonTreeClassification	86.03663392620122
# =============================================================================


# =============================================================================
# Accuracy Model 3:
# 'condition','color_type','X1', 'X2', 'dif'
# =============================================================================
# =============================================================================
# KNN	89.00982213963367
# Logistic Regression	87.92142288293071
# SVC	87.8948765595965
# DecisonTreeClassification	87.20467215290682
# Naive Bayes	36.740111494558
# =============================================================================
# =============================================================================
# KNN	89.62038757632068
# Logistic Regression	87.92142288293071
# SVC	87.8948765595965
# DecisonTreeClassification	87.20467215290682
# Naive Bayes	36.740111494558
# =============================================================================
