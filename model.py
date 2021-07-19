# -*- coding: utf-8 -*-
"""
Created on Mon Jul 19 14:13:37 2021

@author: Sushilkumar
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
import pickle

np.set_printoptions(suppress=True)
 #Read CSV
insuranceDF = pd.read_csv('insurance.csv')
print(insuranceDF.head(5))
insuranceDF.info()
insuranceDF.shape
maxClm = insuranceDF['charges'].max()
minchrg=insuranceDF['charges'].min()
print("Maximum value in column 'charges': ",maxClm )
print("Maximum value in column 'charges': ",minchrg )
# Train Test Split
dfTrain = insuranceDF[:1000]
dfTest = insuranceDF[1000:1300]
dfCheck = insuranceDF[1300:]
# Convert to numpy array
trainLabel = np.asarray(dfTrain['insuranceclaim'])
trainData = np.asarray(dfTrain.drop('insuranceclaim',1))
testLabel = np.asarray(dfTest['insuranceclaim'])
testData = np.asarray(dfTest.drop('insuranceclaim',1))
# Convert to Logistic regressionmodel
insuranceCheck = LogisticRegression()
insuranceCheck.fit(trainData, trainLabel)
accuracy = insuranceCheck.score(testData, testLabel)
print("accuracy = ", accuracy * 100, "%")

# Saving model to file
pickle.dump(insuranceCheck, open('health_insc_model.pkl','wb'))
# Loading model to compare the results
health_insc_model = pickle.load(open('health_insc_model.pkl','rb'))

#Prediction testing

list_a=trainData[2]
print('INPUT LIST:::["age","sex","bmi","children","smoker","region","charges"]')
print("INPUT LIST:::",list_a)
number_of_elements = len(list_a)
print("Total input size:::",number_of_elements)
results=health_insc_model.predict([list_a])
results[0]
print("Health insurance CLAIMS OR NOT:::",results[0])
final=results[0]
if final==0:
    print("SORRY !!!! Insurance has not claim")
elif final==1:
    print("Congratulations !!!! Insurance has claim")