import os
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.externals import joblib
from skimage.io import imread
from skimage.filters import threshold_otsu
from sklearn.model_selection import train_test_split

import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt


from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn import datasets, neighbors, linear_model


train = pd.read_csv('hundreds.csv')


# def encode(train):
#     le = LabelEncoder().fit(train.Labels)
#     labels = le.transform(train.Labels)

#     classes = list(le.classes_)
    
#     train = train.drop(['Labels', 'id'], axis=1)
    
#     return train, labels, classes

# train, labels, classes = encode(train)

# print(train.head(1))

# # print(labels)

# # print(classes)

# ### Stratified test train split 

# sss = StratifiedShuffleSplit(labels, 10, test_size=0.2, random_state=23)

# for train_index, test_index in sss:
#     X_train, X_test = train.values[train_index], train.values[test_index]
#     y_train, y_test = labels[train_index], labels[test_index]


Y_train = train["Labels"]

# Drop 'label' column
X_train = train.drop(labels = ["Labels", "id"],axis = 1)

# Y_train = to_categorical(Y_train, num_classes = 10)
random_seed = 2

X_train, X_test, y_train, y_test = train_test_split(X_train, Y_train, test_size = 0.1, random_state=random_seed) 


knn = neighbors.KNeighborsClassifier()
# logistic = linear_model.LogisticRegression()
knn_model = knn.fit(X_train, y_train)

print(knn_model.score(X_test, y_test))

current_dir = os.path.dirname(os.path.realpath(__file__))
save_directory = os.path.join(current_dir, 'models/knn/')
if not os.path.exists(save_directory):
    os.makedirs(save_directory)
joblib.dump(knn_model, save_directory+'/knn.pkl')
