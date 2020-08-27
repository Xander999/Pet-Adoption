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
# This is a children of breed.py file. In this we will be using Hyperparameters
# tuning to get highest accuracy out of KNN and Xboost
# =============================================================================




# =============================================================================
# #---------------Fitting Decision Tree Classification to the Training set------
# =============================================================================
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
DTC_cl_breed =  DecisionTreeClassifier(criterion = 'gini', random_state = 1234)

#Hyper Parameters Set
params = {'max_features': ['auto', 'sqrt', 'log2'],
          'min_samples_split': [2,3,4,5,6,7,8,9,10,11,12,13,14,15], 
          'min_samples_leaf':[1,2,3,4,5,6,7,8,9,10,11],
          'random_state':[123]}
#Making models with hyper parameters sets
DTC_cl_breed11 = GridSearchCV(DTC_cl_breed, param_grid=params, n_jobs=-1)
#Learning
DTC_cl_breed11.fit(X_train11, Y_train11['breed_category'])
#The best hyper parameters set
print("Best Hyper Parameters for Decison Tree:",DTC_cl_breed11.best_params_)

# Predicting the Test set results
Y_pred11_breed_category = DTC_cl_breed11.predict(X_test11)

# Making the Confusion Matrix and Accuracy Score
from sklearn.metrics import confusion_matrix, accuracy_score
DTC_cm_breed11 = confusion_matrix(Y_test11['breed_category'], Y_pred11_breed_category)
DTC_acc_breed11=accuracy_score(Y_test11['breed_category'], Y_pred11_breed_category)

# =============================================================================
# Best Hyper Parameters for Decison Tree: {'max_features': 'auto', 
#                                          'min_samples_leaf': 2, 
#                                          'min_samples_split': 13, 
#                                          'random_state': 123}
#  DTC_acc_breed11
# Out[6]: 0.9145208388638174
# =============================================================================







# =============================================================================
# -------------------Fitting Xboost to the Training Set--------------------
# =============================================================================
import xgboost as xgb
from sklearn.model_selection import GridSearchCV


xg_reg = xgb.XGBClassifier(objective ='reg:logistic', n_estimators=350, seed=30)
                           
#Hyper Parameters Set
params = {'colsample_bytree': [0.5,0.6,0.7],
          'learning_rate': [0.4,0.5,0.6], 
          'gamma':[0.1,0.2],
          'subsample':[0.8,0.9],
          'max_depth':[4,5,6],
          'min_child_weight':[3,4,5,6],
          'alpha':[12,13,14,15]}

#Making models with hyper parameters sets
xg_reg11 = GridSearchCV(xg_reg, param_grid=params, n_jobs=-1)
#Learning
xg_reg11.fit(X_train11, Y_train11['breed_category'])

Y_pred11_breed_category = xg_reg11.predict(X_test11)
#The best hyper parameters set
print("Best Hyper Parameters for Decison Tree:",xg_reg11.best_params_)


# Making the Confusion Matrix and Accuracy Score
from sklearn.metrics import confusion_matrix, accuracy_score
XBOOST_cm_breed11 = confusion_matrix(Y_test11['breed_category'], Y_pred11_breed_category)
XBOOST_acc_breed11=accuracy_score(Y_test11['breed_category'], Y_pred11_breed_category)

