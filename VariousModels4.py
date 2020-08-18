#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  9 17:32:31 2020

@author: xander999
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def xx(x):
    x=x.strip()
    return x.split(' ')
# =============================================================================
# Fetching the Data
# =============================================================================

dataset=pd.read_csv('Dataset/train.csv')

# =============================================================================
# Here we are ignoring the condition coloumn as it contains null value 
# =============================================================================

data=dataset.loc[:,['pet_id','color_type','length(m)',
                   'height(cm)', 'X1', 'X2', 'breed_category',
                   'pet_category']]
data=data.set_index('pet_id')
# =============================================================================
# #---------Missing Value------------As the particular coloumn is missing 
# ------we removedd the code.
# =============================================================================

data['color_type']=data['color_type'].apply(xx)
X_train1=data.loc[:,['color_type','length(m)',
                   'height(cm)', 'X1', 'X2']]
Y_train1=data.loc[:,['breed_category', 'pet_category']]


# =============================================================================
# #---------Categorical Values---------
# =============================================================================
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
knn_cl_pet = KNeighborsClassifier(n_neighbors =4, metric = 'minkowski', p = 2)


#------For X_train11 and Y_train11-------------
knn_cl_breed.fit(X_train11, Y_train11['breed_category'])
knn_cl_pet.fit(X_train11, Y_train11['pet_category'])

# Predicting the Test set results
Y_pred11_breed_category = knn_cl_breed.predict(X_test11)
Y_pred11_pet_category = knn_cl_pet.predict(X_test11)

# Making the Confusion Matrix and Accuracy Score
from sklearn.metrics import confusion_matrix, accuracy_score
knn_cm_breed11 = confusion_matrix(Y_test11['breed_category'], Y_pred11_breed_category)
knn_acc_breed11=accuracy_score(Y_test11['breed_category'], Y_pred11_breed_category)

knn_cm_pet11 = confusion_matrix(Y_test11['pet_category'], Y_pred11_pet_category)
knn_acc_pet11=accuracy_score(Y_test11['pet_category'], Y_pred11_pet_category)






# =============================================================================
# #-------------------Fitting Logistic Regression to the Training set----------
# =============================================================================
from sklearn.linear_model import LogisticRegression
logReg_cl_breed = LogisticRegression(random_state = 0)
logReg_cl_pet = LogisticRegression(random_state=0)


#------For X_train11 and Y_train11-------------
logReg_cl_breed.fit(X_train11, Y_train11['breed_category'])
logReg_cl_pet.fit(X_train11, Y_train11['pet_category'])

# Predicting the Test set results
Y_pred11_breed_category = logReg_cl_breed.predict(X_test11)
Y_pred11_pet_category = logReg_cl_pet.predict(X_test11)

# Making the Confusion Matrix and Accuracy Score
from sklearn.metrics import confusion_matrix, accuracy_score
log_cm_breed11 = confusion_matrix(Y_test11['breed_category'], Y_pred11_breed_category)
log_acc_breed11=accuracy_score(Y_test11['breed_category'], Y_pred11_breed_category)

log_cm_pet11 = confusion_matrix(Y_test11['pet_category'], Y_pred11_pet_category)
log_acc_pet11=accuracy_score(Y_test11['pet_category'], Y_pred11_pet_category)








# =============================================================================
# #-----------------------Fitting SVM to the Training set---------------------
# =============================================================================
from sklearn.svm import SVC
SVC_cl_breed = SVC(kernel = 'linear', random_state = 0)
SVC_cl_pet = SVC(kernel = 'linear', random_state = 0)


#------For X_train11 and Y_train11-------------
SVC_cl_breed.fit(X_train11, Y_train11['breed_category'])
SVC_cl_pet.fit(X_train11, Y_train11['pet_category'])

# Predicting the Test set results
Y_pred11_breed_category = SVC_cl_breed.predict(X_test11)
Y_pred11_pet_category = SVC_cl_pet.predict(X_test11)

# Making the Confusion Matrix and Accuracy Score
from sklearn.metrics import confusion_matrix, accuracy_score
SVC_cm_breed11 = confusion_matrix(Y_test11['breed_category'], Y_pred11_breed_category)
SVC_acc_breed11=accuracy_score(Y_test11['breed_category'], Y_pred11_breed_category)

SVC_cm_pet11 = confusion_matrix(Y_test11['pet_category'], Y_pred11_pet_category)
SVC_acc_pet11=accuracy_score(Y_test11['pet_category'], Y_pred11_pet_category)








# =============================================================================
# # ----------------Fitting Naive Bayes to the Training set-----------------------
# =============================================================================
from sklearn.naive_bayes import GaussianNB
Gaus_cl_breed =  GaussianNB()
Gaus_cl_pet =  GaussianNB()


#------For X_train11 and Y_train11-------------
Gaus_cl_breed.fit(X_train11, Y_train11['breed_category'])
Gaus_cl_pet.fit(X_train11, Y_train11['pet_category'])

# Predicting the Test set results
Y_pred11_breed_category = Gaus_cl_breed.predict(X_test11)
Y_pred11_pet_category = Gaus_cl_pet.predict(X_test11)

# Making the Confusion Matrix and Accuracy Score
from sklearn.metrics import confusion_matrix, accuracy_score
Gaus_cm_breed11 = confusion_matrix(Y_test11['breed_category'], Y_pred11_breed_category)
Gaus_acc_breed11=accuracy_score(Y_test11['breed_category'], Y_pred11_breed_category)

Gaus_cm_pet11 = confusion_matrix(Y_test11['pet_category'], Y_pred11_pet_category)
Gaus_acc_pet11=accuracy_score(Y_test11['pet_category'], Y_pred11_pet_category)






# =============================================================================
# #---------------Fitting Decision Tree Classification to the Training set------
# =============================================================================
from sklearn.tree import DecisionTreeClassifier
DTC_cl_breed =  DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
DTC_cl_pet =  DecisionTreeClassifier(criterion = 'entropy', random_state = 0)


#------For X_train11 and Y_train11-------------
DTC_cl_breed.fit(X_train11, Y_train11['breed_category'])
DTC_cl_pet.fit(X_train11, Y_train11['pet_category'])

# Predicting the Test set results
Y_pred11_breed_category = DTC_cl_breed.predict(X_test11)
Y_pred11_pet_category = DTC_cl_pet.predict(X_test11)

# Making the Confusion Matrix and Accuracy Score
from sklearn.metrics import confusion_matrix, accuracy_score
DTC_cm_breed11 = confusion_matrix(Y_test11['breed_category'], Y_pred11_breed_category)
DTC_acc_breed11=accuracy_score(Y_test11['breed_category'], Y_pred11_breed_category)

DTC_cm_pet11 = confusion_matrix(Y_test11['pet_category'], Y_pred11_pet_category)
DTC_acc_pet11=accuracy_score(Y_test11['pet_category'], Y_pred11_pet_category)



# =============================================================================
# After Evaluation of 5 different Models we have accuracy for each models  which 
# we will make on the DataFrame nameing "Accurracy_model". And then we can display
# it on the graph.
# =============================================================================



breed_acc=[knn_acc_breed11*100, log_acc_breed11*100, SVC_acc_breed11*100, Gaus_acc_breed11*100, DTC_acc_breed11*100]
pet_acc= [knn_acc_pet11*100, log_acc_pet11*100, SVC_acc_pet11*100, Gaus_acc_pet11*100, DTC_acc_pet11*100]
rows=['KNN', 'Logistic Regression', 'SVC', 'Naive Bayes', 'DecisonTreeClassification']

Accuracy_model=pd.DataFrame(data={'Breed':breed_acc, 'Pet': pet_acc}, index=rows)

ax=Accuracy_model.plot.bar(figsize=(12,9), rot=0, bottom=81)




# =============================================================================
# From the evaluation from various models we come at a conclusion that the 
# for classification of breed Category we need Decion Tree Classification
# and for classification of pet Category we need either Support Vector 
#Classification or Logistic Regression Classification. 
# =============================================================================
