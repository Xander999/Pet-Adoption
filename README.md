# Pet-Adoption
This an Hacker Earath sponsored Maachine Learning Challenge.
---------------Pet Adoption Challenege-----------

The File system of this current folder consists of the
1.Dataset(Folder)
2.Other Final Models(Folder)
3.Various Models(Folder)
4.breed.py
5.breed1.py
6.pet.py
7.pet1.py
8.process.py
9.FinalModel5.py  <--- The actual file that consist of final model of my submission
10.Result5.csv  

IDE used: Spyder(Anaconda Navigator)

----------------Process-------------------------------

1. The initial modelling starts by preprocessing of data in the
dataset. We start by fetching data, then converting the time in 
the issue date and listing date columns to a suitable format. 
As the date perform no function in classification we have taken the 
date difference between listing and issue date, and concatenated the
column in another column naming 'dif'.
2. We observed that there is missing value in the 'condition' column.
We tested replacong the nan value with mode and mean value of the column. 
Later discovered all the NaN values represent to another set
of class. Thus we moved forward with mean giving much more accuracy.
Instead of mean we could have replace with any other categorical
numerical value, as the type is categorical it fulfills the
same purpose.
3. We have made dummmy variables for 'condition' and 'color type'
column. For 'condition' column there would have four more variables.
But for 'color type' we have made dummies based on two criteria.
One is we take the whole color code as one value and then convert it 
into multivariate binary variables as a whole, and secondly we can tokenize 
each color code in cells of column 'color type' and then convert it into
mutivariate binary variables.

All the above experiments have been done and been tested on each model such as
logistic classification
KNN Classification 
SVM Classification
Naive Bayes 
Decison Trees and
XGBoost
This experiments have been stored in "Various Models" Folder. With each "various models
a final model have been generated stored in "Other Final Models" Folder.

The second approach of tokenization seems giving better results.

4. Followed by this we have performed normalization by feature scaling.

5.------------Feature Engineering---------- 
The length(cm) and height(cm) seems to impart no contribution in the classification problem.
Thus we have reemoved it.
Pet.py and Breed.py have been created for this feature engineeing purpose where we have taken various
combination of independent variables for the finding the maximum accuracy. After a long struggle,
we came to the conslusion that for classification of Breed type we need Random Forest and XGboost and
for the classification of Pet Type KNN and XGBoost gives significant results.

6.Pet1.py and Breed1.py is the result of previous step. Where we have performed Hyperparameter Tunning for 
better result. we have applied GridSearch to find optimal tuning for each of the two models for two different
classification. ANd at final step we have used XGBoost for the classification of Pet Type and Random Forest 
for the classification of Breed Type.

-------Conslusion--------    
The final model have been named as "FinalModel5.py" and it saves the result as 'Result5.csv'. 
