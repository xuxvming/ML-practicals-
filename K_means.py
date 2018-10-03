# -*- coding: utf-8 -*-
"""
Created on Fri Jun  8 14:44:51 2018

@author: xxiu
"""
'''
handling non-numerical values using k means clusetring
'''
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import style
from sklearn.cluster import KMeans
style.use('ggplot')
from sklearn import preprocessing, cross_validation
import pandas as pd

df = pd.read_excel('titanic.xls')


#df.drop(['body','name'],1, inplace = True)
#df.convert_object(convert_numeric = True)
df.fillna(0, inplace = True)

def handle_non_numerical_data(df):
    columns = df.columns.values
    
    for column in columns:
        #creating a dictionary with text and numerical values
        text_digit_vals = {}
        def convert_to_int(val):
            #for every column, create a function to convert that into integer values
            return text_digit_vals[val]
        
        #determin the value of each column, if not integer or float. it is char
        if df[column].dtype != np.int64 and df[column].dtype!=np.float64:
            #getting all unique non repteative values 
            column_contents = df[column].values.tolist()
            unique_elements = set(column_contents)
            
            x= 0
            #if the unique value is not in the dictionary, put it in
            for unique in unique_elements:
                if unique not in text_digit_vals:
                    text_digit_vals[unique] = x
                    x+=1
            #reseting the comlumn by mapping the function with that column       
            df[column] = list(map(convert_to_int,df[column]))
            
            
    return df

df= handle_non_numerical_data(df)
print(df.head())


df.drop(['sex','boat'],1,inplace = True)
X= np.array(df.drop(['survived'],1).astype(float))
X = preprocessing.scale(X)
y= np.array(df['survived'])
 
clf = KMeans(n_clusters = 2)
clf.fit(X)

correct = 0
for i in range(len(X)):
    predict_me = np.array(X[i].astype(float))
    predict_me = predict_me.reshape(-1,len(predict_me))
    prediction = clf.predict(predict_me)
    if prediction[0] == y[i]:
        correct +=1

print(correct/len(X))













