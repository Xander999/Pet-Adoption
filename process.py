#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  8 23:05:13 2020

@author: xander999
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# =============================================================================
# Fetching the Data
# =============================================================================

dataset=pd.read_csv('Dataset/train.csv')

data=dataset.loc[:,['pet_id','condition','color_type','length(m)',
                   'height(cm)', 'X1', 'X2', 'breed_category',
                   'pet_category']]
data=data.set_index('pet_id')
# =============================================================================
# X_train=dataset.loc[:,['pet_id','condition','color_type','length(m)',
#                    'height(cm)', 'X1', 'X2']]
# X_train=X_train.set_index('pet_id')
# 
# Y_train=dataset.loc[:,['pet_id', 'breed_category', 'pet_category']]
# Y_train=Y_train.set_index('pet_id')
# =============================================================================




# =============================================================================
# We will be using various classification models and research which 
# classification models fits best on the "trrain.csv" dataset file. 
# =============================================================================





# =============================================================================
# Preprocessing of Data
#
# In this step we can see that null or nan values are constituted only in
# 'coondition' column having 1477 null rows. So we can do two steps out of this:
#     1. Remopve the rows having null values in condition coloumn
#     2. Or we rreplace it with most frequent values as condition column is
#       is of categorical type.
# =============================================================================

print(len(data[data['condition'].isna()]))

#---------Missing Value------------
# X_train1 contains values with drropped null values
#X_train1 = X_train[~X_train['condition'].isna()]
data1=data[~data['condition'].isna()]
X_train1=data1.loc[:,['condition','color_type','length(m)',
                   'height(cm)', 'X1', 'X2']]
Y_train1=data1.loc[:,['breed_category', 'pet_category']]


# X_train2 contains values with replacement of null values with most frrequent
#X_train2 = X_train.fillna(X_train.mode().iloc[0])
data2=data.fillna(data.mode().iloc[0])
X_train2=data2.loc[:,['condition','color_type','length(m)',
                   'height(cm)', 'X1', 'X2']]
Y_train2=data2.loc[:,['breed_category', 'pet_category']]


#---------Categorical Values---------
X_train1=pd.get_dummies(X_train1, prefix=['Condition_'], columns=['condition'])
X_train1=pd.get_dummies(X_train1, prefix=['Color_'], columns=['color_type'])

X_train2=pd.get_dummies(X_train2, prefix=['Condition_'], columns=['condition'])
X_train2=pd.get_dummies(X_train2, prefix=['Color_'], columns=['color_type'])


#---------Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train11, X_test11, Y_train11, Y_test11 = train_test_split(X_train1, Y_train1, test_size = 0.2, random_state = 0)
X_train22, X_test22, Y_train22, Y_test22 = train_test_split(X_train2, Y_train2, test_size = 0.2, random_state = 0)


#---------Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train11 = sc_X.fit_transform(X_train11)
X_test11 = sc_X.transform(X_test11)
X_train22 = sc_X.fit_transform(X_train22)
X_test22 = sc_X.transform(X_test22)


# =============================================================================
# Analysis
# =============================================================================


f1=dataset['condition'].value_counts()
f2=dataset['color_type'].value_counts()
f3=dataset['length(m)'].value_counts()
f4=dataset['height(cm)'].value_counts()
f5=dataset['X1'].value_counts()
f6=dataset['X2'].value_counts()

f1x=(dataset[['breed_category','pet_category']].groupby(dataset['condition']).count())
f2x=(dataset[['breed_category','pet_category']].groupby(dataset['color_type']).count())
f3x=(dataset[['breed_category','pet_category']].groupby(dataset['length(m)']).count())
f4x=(dataset[['breed_category','pet_category']].groupby(dataset['height(cm)']).count())
f5x=(dataset[['breed_category','pet_category']].groupby(dataset['X1']).count())
f6x=(dataset[['breed_category','pet_category']].groupby(dataset['X2']).count())




# =============================================================================
# 
# After ppreprocessing and  analysis state... we will be using different 
# classification models such as KNN, SVM, Logistic Regrression and Naive Bayes.
# =============================================================================

# =============================================================================
# We will be using two different classifier for identifiication of breed and
# pet for each differentr datasets i.e, X_train11 and X_train22
# =============================================================================

# Fitting K-NN to the Training set
from sklearn.neighbors import KNeighborsClassifier

cl_breed = KNeighborsClassifier(n_neighbors = 3, metric = 'minkowski', p = 2)
cl_pet = KNeighborsClassifier(n_neighbors =4, metric = 'minkowski', p = 2)

#------For X_train11 and Y_train11-------------
cl_breed.fit(X_train11, Y_train11['breed_category'])
cl_pet.fit(X_train11, Y_train11['pet_category'])

# Predicting the Test set results
Y_pred11_breed_category = cl_breed.predict(X_test11)
Y_pred11_pet_category = cl_pet.predict(X_test11)

# Making the Confusion Matrix and Accuracy Score
from sklearn.metrics import confusion_matrix, accuracy_score
cm_breed11 = confusion_matrix(Y_test11['breed_category'], Y_pred11_breed_category)
acc_breed11=accuracy_score(Y_test11['breed_category'], Y_pred11_breed_category)
cm_pet11 = confusion_matrix(Y_test11['pet_category'], Y_pred11_pet_category)
acc_pet11=accuracy_score(Y_test11['pet_category'], Y_pred11_pet_category)



#------For X_train22 and Y_train22-------------
cl_breed.fit(X_train22, Y_train22['breed_category'])
cl_pet.fit(X_train22, Y_train22['pet_category'])

# Predicting the Test set results
Y_pred22_breed_category = cl_breed.predict(X_test22)
Y_pred22_pet_category = cl_pet.predict(X_test22)

# Making the Confusion Matrix and Accuracy Score
from sklearn.metrics import confusion_matrix, accuracy_score
cm_breed22 = confusion_matrix(Y_test22['breed_category'], Y_pred22_breed_category)
acc_breed22 =accuracy_score(Y_test22['breed_category'], Y_pred22_breed_category)
cm_pet22 = confusion_matrix(Y_test22['pet_category'], Y_pred22_pet_category)
acc_pet22 =accuracy_score(Y_test22['pet_category'], Y_pred22_pet_category)


# =============================================================================
# 
# print(acc_breed22,'  ', acc_pet22)
# 0.848420493761614    0.8316963100610566
# 
# print(acc_breed11,'  ', acc_pet11)
# 0.8827764976958525    0.8352534562211982
# 

We can see from above observation that for KNN if we drop the null values produces
more accuracy than the replacing it with the most frequent value.......
# =============================================================================
