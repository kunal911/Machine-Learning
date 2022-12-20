# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 12:44:22 2021

@author: Spencer Peterson
"""

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
from sklearn.metrics import accuracy_score

train = 'income2022f/train_final.csv'
test = 'income2022f/test_final.csv'


df = pd.read_csv(train)
# %%Change numeric values from range to true or false and handle unknowns

def ProcessTrainingDataFrame(df):
         # Check if each column is all numeric
    for colInd in range(0,len(df.columns)-1):
        # if the column is numeric, split into below and above median as a boolean
    
        if pd.to_numeric(df.iloc[:,colInd], errors='coerce').notnull().all():
            #print(colInd, "is all numeric")
            df.iloc[:,colInd] = (df.iloc[:,colInd] <= df.iloc[:,colInd].median())
    
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
         # Check if each column is all numeric
    for colInd in range(1,len(df.columns)):
        # if the column is numeric, split into below and above median as a boolean
    
        if pd.to_numeric(df.iloc[:,colInd], errors='coerce').notnull().all():
            #print(colInd, "is all numeric")
            df.iloc[:,colInd] = (df.iloc[:,colInd] <= df.iloc[:,colInd].median())
    
    df = df.astype(object)


    for colInd in range(0,len(df.columns)):
        currMode = df[df.columns[colInd]].mode()[0]
        df[df.columns[colInd]].replace({'?' : currMode}, inplace=True)
    
    
    return df

# %% 

df = ProcessTrainingDataFrame(df)
labelProcessor = preprocessing.LabelEncoder()

for i in range(14):
    df.iloc[:,i] = labelProcessor.fit_transform(df.iloc[:,i])

# %% 

Y = df.iloc[:,-1]
X = df.iloc[:,0:14]

Y = Y.astype(int)
neighbors = KNeighborsClassifier(n_neighbors=10, weights='distance')
neighbors.fit(X,Y)

test_predictions = neighbors.predict(X)
print(accuracy_score(Y, test_predictions))


testDF = pd.read_csv(test)
testDF = ProcessTestingDataFrame(testDF)

for i in range(1,15):
    testDF.iloc[:,i] = labelProcessor.fit_transform(testDF.iloc[:,i])



predictions = neighbors.predict(testDF.iloc[:,1:15])

predictionDF = pd.DataFrame(predictions)

predictionDF["ID"] = testDF["ID"].values
predictionDF.to_csv('Predictions_KNN.csv', index=False, header=['Prediction','ID'])