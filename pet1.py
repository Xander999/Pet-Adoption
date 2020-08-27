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
# This is a children of pet.py file. In this we will be using Hyperparameters
# tuning to get highest accuracy out of KNN and Xboost
# =============================================================================




# =============================================================================
# #-------------------Fitting K-NN to the Training set---------------------
# =============================================================================
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
knn_cl_pet = KNeighborsClassifier(n_jobs=-1, metric = 'minkowski', p = 2)

#Hyper Parameters Set
params = {'n_neighbors':[5,6,7,8,9,10],
          'leaf_size':[1,2,3,5],
          'weights':['uniform', 'distance'],
          'algorithm':['auto', 'ball_tree','kd_tree','brute'],
          'n_jobs':[-1]}
#Making models with hyper parameters sets
knn_cl_pet11 = GridSearchCV(knn_cl_pet, param_grid=params, n_jobs=1)

#------For X_train11 and Y_train11-------------
knn_cl_pet11.fit(X_train11, Y_train11['pet_category'])

# Predicting the Test set results
Y_pred11_pet_category = knn_cl_pet11.predict(X_test11)

# Making the Confusion Matrix and Accuracy Score
from sklearn.metrics import confusion_matrix, accuracy_score
knn_cm_pet11 = confusion_matrix(Y_test11['pet_category'], Y_pred11_pet_category)
knn_acc_pet11=accuracy_score(Y_test11['pet_category'], Y_pred11_pet_category)

#The best hyper parameters set
print("Best Hyper Parameters for Decison Tree:",knn_cl_pet11.best_params_)


# =============================================================================
# Tree:",knn_cl_pet11.best_params_)
# Best Hyper Parameters for Decison Tree: {'algorithm': 'kd_tree', 'leaf_size': 3, 'n_jobs': -1, 'n_neighbors': 7, 'weights': 'uniform'}
# 
# knn_acc_pet11
# Out[5]: 0.8914255375630475
# =============================================================================


# =============================================================================
# -------------------Fitting Xboost to the Training Set--------------------
# =============================================================================
import xgboost as xgb
xg_reg = xgb.XGBClassifier(objective ='reg:logistic',
                           colsample_bytree=0.6,
                           learning_rate=0.5, 
                           gamma=0.1, 
                           subsample=0.9,
                           seed=40,
                           n_estimators=390,
                           max_depth=5,
                           min_child_weight=5,
                           alpha=13)

#Making models with hyper parameters sets
xg_reg11 = GridSearchCV(xg_reg, param_grid=params, n_jobs=-1)
#Learning
xg_reg.fit(X_train11, Y_train11['pet_category'])

Y_pred11_pet_category = xg_reg.predict(X_test11)

# Making the Confusion Matrix and Accuracy Score
from sklearn.metrics import confusion_matrix, accuracy_score
XBOOST_cm_pet11 = confusion_matrix(Y_test11['pet_category'], Y_pred11_pet_category)
XBOOST_acc_pet11=accuracy_score(Y_test11['pet_category'], Y_pred11_pet_category)
print(XBOOST_acc_pet11*100)


# =============================================================================
# 90.23095301300769
# =============================================================================
