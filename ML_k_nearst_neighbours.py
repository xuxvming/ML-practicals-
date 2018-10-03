# -*- coding: utf-8 -*-
"""
Created on Mon Jun  4 19:46:04 2018

@author: oliver
"""
#KNN The value of K is often taken as a odd number
# euclidian distance 

import numpy as np
from sklearn import preprocessing, cross_validation,neighbors
import pandas as pd

df = pd.read_csv("breast-cancer-wisconsin.data.txt")
#replace missing values with -99999
df.replace("?", -99999, inplace = True);
#droping id column which is useless
df.drop(["id"],1,inplace = True);

#feature is every thing but class
X = np.array(df.drop(["class"],1));
#label is class
y = np.array(df["class"]);

#cross_validation, training
X_train, X_test,y_train,y_test = cross_validation.train_test_split(X,y,test_size = 0.2);

#define classifier
clf = neighbors.KNeighborsClassifier();
clf.fit(X_train,y_train);

accuracy= clf.score(X_test,y_test);
print ("the accuracy is:");
print(accuracy);

#making an example to predict
example_measure = np.array([4,2,1,1,1,2,3,2,1]);
example_measure = example_measure.reshape(len(example_measure),-1);

prediction = clf.predict(example_measure);
print(prediction);
