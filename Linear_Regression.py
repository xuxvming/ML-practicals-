# -*- coding: utf-8 -*-

"""
Created on Sun May 27 01:55:13 2018

@author: oliver
"""

# Concepts:
# feature: the input for training data
# lable: the output getting from the model after training 
# Linear regression: fiting a line

import pandas as pd
import quandl
import math, datetime
import numpy as np 
#allow us to use arrays 
from sklearn import preprocessing ,cross_validation,svm
#using scaling, help with accuracy and speed up the calculation
#to create training and testing sample so that we won't get biased sample
#support vector machine ,regression
from sklearn.linear_model import LinearRegression 
import matplotlib.pyplot as plt
from matplotlib import style
import pickle
#plot the graph
style.use('ggplot')


df = quandl.get("WIKI/GOOGL")
df = df[['Adj. Open','Adj. High','Adj. Low','Adj. Close','Adj. Volume']]

df['HL_PCT'] = (df['Adj. High']-df['Adj. Close'])/df['Adj. Close']
df['PCT_change'] = (df['Adj. Close']-df['Adj. Open'])/df['Adj. Open']

df = df[['Adj. Close','HL_PCT','PCT_change','Adj. Volume']]

#forecast column
forecast_col = 'Adj. Close'

#filter the data with bad values
df.fillna('-9999',inplace = True)

#math.ceil round up by 1 
#trying to predict 1% of the data i.e. 10 days in the future 
forecast_out = int(math.ceil(0.01*len(df)))
print(forecast_out)
#shifting the columns up
#the label for each row will be 10 days in the future
df['label'] = df[forecast_col].shift(-forecast_out)


#X is our feature 
X = np.array(df.drop(['label'],1))
#scale the new values but also scale it alongside with other values 
X = preprocessing.scale(X)
# we don't have the y values for X_lately so that's what we train for getting 
X_lately = X[-forecast_out:]
X= X[:-forecast_out]

#The values of X are 3389 and have corresponding values of y
#We are trying to predict the Values of next 35s of ys given Xs 



#y is our label
#dropping missing values 
df.dropna(inplace=True)
y = np.array(df['label'])


#creating a training and testing set
#20% of the data will be used as test data
X_train,X_test,y_train,y_test = cross_validation.train_test_split(X,y,test_size = 0.2)

#fit the regression line using train value 
#check the document to find more parameters 
clf = LinearRegression(n_jobs=-1);
#swicth to support vector regression algorithm
clf =svm.SVR();
#fit the classfier, the clf will be used to predict the future 
clf.fit(X_train,y_train)

#pickle allows to save the model
with open('linerrageression.pickle','wb') as f:
    pickle.dump(clf, f);

#reload the saved model 
pickle_in = open('linerrageression.pickle','rb');
clf = pickle.load(pickle_in);

#test the classfier
accuracy = clf.score(X_test,y_test)

forecast_set = clf.predict(X_lately);
print(forecast_set,accuracy,forecast_out);

#specify the entire column is just full of not a number data 
df["forecast"] = np.nan;

#find the last day
last_date = df.iloc[-1].name;
last_unix = last_date.timestamp();
one_day = 86400
next_unix = last_unix +one_day;

print(df.ix[2900]);
#populating the date frame with new date values
#X and Y doesn't necessarily the axes on the graph 
#X is features and Y is the label, it just heppens that the label id the price
#X is not correct because the date is not a feature
for i in forecast_set:
    next_date =datetime.datetime.fromtimestamp(next_unix);
    next_unix += one_day;
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)]+[i];

print(df.tail());

df["Adj. Close"].plot();
df["forecast"].plot();
plt.legend(loc=4);
plt.xlabel("Date");
plt.ylabel("Price");
plt.show();