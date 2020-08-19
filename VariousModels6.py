#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 12:00:19 2020

@author: xander999
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# #---- This function tokenizes different spaced 'color_type' in the dataset-- 
# def xx(x):
#     x=x.strip()
#     return x.split(' ')
# 
# =============================================================================
# =============================================================================
# Fetching the Data
# =============================================================================

dataset=pd.read_csv('Dataset/train.csv')

data=dataset.loc[:,['pet_id','condition','color_type','length(m)',
                   'height(cm)', 'X1', 'X2', 'breed_category',
                   'pet_category']]
data=data.set_index('pet_id')


# =============================================================================
 # Instead of replcaing the missing values with the mode of the 'condition' 
 # variable we will try to predict those value first from the variable 
 # 'color_type', 'X1' and 'X2'. We assume color of certain pet is solely dedpendent on the
 # condition of the pet. Thus, haaving the hypothesis we move forward.
 # 
 # We will make two approach out of it.
 # First we will tokenize the color coded of each pet and try to binary the value
 # by multivariate dummy variable.
 # 
 # Secondly we will make dummies without any tokenization. 
# =============================================================================


dfd=dataset.loc[:,['pet_id', 'condition','color_type','X1','X2']]

# =============================================================================
# Second Approach
# =============================================================================
train2=dfd[~dfd['condition'].isna()]
train2=train2.set_index('pet_id')
train2=pd.get_dummies(train2, prefix=[''], columns=['color_type'])

train2_x=train2.iloc[:,1:]
train2_y=train2.loc[:,['condition']]

pred_x=dfd[dfd['condition'].isna()]
pred_x=pred_x.set_index('pet_id')
pred_x=pred_x.drop('condition', axis=1)
pred_x=pd.get_dummies(pred_x, prefix=[''], columns=['color_type'])


# =============================================================================
# The color_type for train_x and pred_x have different colorr combination.
# In order to predict it we need a common color_type dummy matrix for both 
# training and predicting independent variables.
# =============================================================================

l1=train2_x.iloc[:,2:].columns
l2=pred_x.iloc[:,2:].columns

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
    train2_x[x]=[0]*len(train2_x.index)
for x in b1:
    pred_x[x]=[0]*len(pred_x.index)
                     
        

from sklearn.neighbors import KNeighborsClassifier
knn_cl_condition1 = KNeighborsClassifier(n_neighbors = 3, metric = 'minkowski', p = 2)
knn_cl_condition1.fit(train2_x, train2_y)
pred_y=knn_cl_condition1.predict(pred_x)

pred_y=pd.DataFrame(pred_y)
pred_y=pred_y.set_index(pred_x.index)


for x in pred_y.index:
    data.at[x, 'condition']=pred_y.at[x, 0].astype(int)
    
def cc(x):
    x=str(x)
    st=x.split(' ')
    return st[0]
data['dif']=pd.Series(pd.to_datetime(dataset.listing_date, format='%Y-%m-%d')
                           -pd.to_datetime(dataset.issue_date, format='%Y-%m-%d')).apply(cc).array
# =============================================================================
# Missing Values are predicted now we will follow the same procedure.
# =============================================================================

X_train1=data.loc[:,['condition','color_type','length(m)',
                   'height(cm)', 'X1', 'X2','dif']]
Y_train1=data.loc[:,['breed_category', 'pet_category']]


# =============================================================================
# #---------Categorical Values---------
# =============================================================================
X_train1=pd.get_dummies(X_train1, prefix=['Condition_'], columns=['condition'])
X_train1=pd.get_dummies(X_train1, prefix=['Color_'], columns=['color_type'])

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

Accuracy_model1=pd.DataFrame(data={'Breed':breed_acc, 'Pet': pet_acc}, index=rows)

ax=Accuracy_model1.plot.bar(figsize=(12,9), rot=0, bottom=81)








