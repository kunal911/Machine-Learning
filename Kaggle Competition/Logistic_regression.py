# -*- coding: utf-8 -*-
"""
Created on Wed May  5 16:13:16 2021

@author: Spencer Peterson
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn.metrics import precision_recall_curve
import os
import sys

train = 'income2022f/train_final.csv'
test = 'income2022f/test_final.csv'


df = pd.read_csv(train)


def ProcessTrainingDataFrame(df):
         # Check if each column is all numeric
    
    df = df.astype(object)
    
    df_yes = df[df[df.columns[14]]==1]
    
    for colInd in range(0,len(df.columns)-1):
        currMode = df_yes[df_yes.columns[colInd]].mode()[0]
        df_yes[df_yes.columns[colInd]].replace({'?' : currMode}, inplace=True)
    
    
    df_no = df[df[df.columns[14]]==0]
        # Change unknowns to most common value in column                
     
    for colInd in range(0,len(df.columns)-1):
        currMode = df_no[df_no.columns[colInd]].mode()[0]
        df_no[df_no.columns[colInd]].replace({'?' : currMode}, inplace=True)
        
        
    return pd.concat((df_yes, df_no))


def ProcessTestingDataFrame(df):
    df = df.astype(object)


    for colInd in range(0,len(df.columns)):
        currMode = df[df.columns[colInd]].mode()[0]
        df[df.columns[colInd]].replace({'?' : currMode}, inplace=True)
    
    
    return df


def ExpandDataframe(df):
    len_cols = len(df.columns)
    
    for i in range(len_cols):
        for j in range(1, len_cols):
            df[len(df.columns)] = df.iloc[:,i] * df.iloc[:,j]

# %%

df = ProcessTrainingDataFrame(df)

#df = df.drop(['native.country', 'education'], axis=1)


labelProcessor = preprocessing.LabelEncoder()

for i in range(len(df.columns)):
    df.iloc[:,i] = labelProcessor.fit_transform(df.iloc[:,i])
    


Y = df.iloc[:,-1]
X = df.iloc[:,0:14]
ExpandDataframe(X)


logReg = LogisticRegression(random_state=0, max_iter=1000)
logReg.fit(X,Y)


test_predictions = logReg.predict_proba(X)

# %% 
print(precision_recall_curve(Y, test_predictions[:,1]))

# %% 
testDF = pd.read_csv(test)
testDF = ProcessTestingDataFrame(testDF)

for i in range(len(testDF.columns)):
    testDF.iloc[:,i] = labelProcessor.fit_transform(testDF.iloc[:,i])
    
test_x = testDF.iloc[:,1:15]
ExpandDataframe(test_x)

predictions = logReg.predict_proba(test_x)

predictionDF = pd.DataFrame(predictions[:,1])
predictionDF["ID"] = testDF["ID"].values
predictionDF["ID"] += 1
predictionDF.to_csv('LogisticRegression.csv', index=False, header=['Prediction','ID'])