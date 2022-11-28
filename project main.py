# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 14:01:00 2021

@author: user
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
data=pd.read_csv('train_ctrUa4K.csv')
print(data)


nullData=data.isna()
print(nullData)
data.info()
missingcount=nullData.sum()
print('missingcount=',missingcount)
data=data.dropna(how='any')
print('b=',data)
import category_encoders as ce
a=data['Loan_Status']
b=pd.DataFrame(a)
print(b)
encoder=ce.OrdinalEncoder(cols='Loan_Status',mapping=[{'col':'Loan_Status','mapping':{'N':0,'Y':1}}])
data1=encoder.fit_transform(b)
print(data1)
data['Loan_Status']=data1['Loan_Status']
print(data)

x=data.drop(['Loan_Status','Loan_ID','Gender','Married','Dependents','Education','Self_Employed','ApplicantIncome','CoapplicantIncome','CoapplicantIncome','Property_Area','Loan_Amount_Term'],axis='columns')
print(x)
print(type(x))
y=data['Loan_Status']
print(y)
#svm
def train():
    x1_train,x1_test,y1_train,y1_test=train_test_split(x,y,test_size=0.25,random_state=40)
    print(x1_train.shape)
    print(x1_test.shape)
    print(y1_train.shape)
    print(y1_test.shape)
    print(type(y1_train))
    from sklearn.svm import SVC
    model1=SVC(C=6,gamma=.1)
    model1.fit(x1_train,y1_train)
    accuracy1=model1.score(x1_test,y1_test)
    print('model1 accuracy=',accuracy1)
    return accuracy1
a=train()


  
#decision tree
from sklearn.tree import DecisionTreeClassifier
x2_train,x2_test,y2_train,y2_test=train_test_split(x,y,test_size=0.25,random_state=40)
print(x2_train.shape)
print(x2_test.shape)
print(y2_train.shape)
print(y2_test.shape)
print(y2_test)
classifier=DecisionTreeClassifier(random_state=40,max_depth=20,min_samples_leaf=20)
classifier.fit(x2_train,y2_train)
ypred=classifier.predict(x2_test)
print(ypred)
#ypred=ypred.reshape(-1,1)
#y2_test=y2_test.reshape(-1,1)
accuracy2=classifier.score(x2_test,y2_test)
print('model2 accuracy=',accuracy2)

#RandomForest
from sklearn.ensemble import RandomForestClassifier
classifier2=RandomForestClassifier(n_estimators=100,max_depth=2)
classifier2.fit(x2_train,y2_train) 
ypred=classifier2.predict(x2_test)
print(ypred)
accuracy3=classifier2.score(x2_test,y2_test)
print('model3 accuracy=',accuracy3)
plt.scatter('svm',a*100,label='svm')
plt.scatter('decision tree',accuracy2*100,label='decision tree')
plt.scatter('RandomForest',accuracy3*100,label='RandomForest')
plt.title('accuracy plot')
plt.xlabel('model')
plt.ylabel('accuracy')
plt.show()
from sklearn.externals import joblib
joblib.dump(classifier,'dect.xml')

testdata=pd.read_csv('test_lAUu6dG.csv')
print(testdata)
testdata=testdata.dropna(how='any')
test=testdata.drop(['Loan_ID'],axis=1)

#print(test)
#a=test['Property_Area']
#b=pd.DataFrame(a)
#encoder=ce.OrdinalEncoder(cols='Property_Area',mapping=[{'col':'Property_Area','mapping':{'Urban':0,'Semiurban':1,'Rural':2}}])
#data1=encoder.fit_transform(b)
#print(data1)
#test['Property_Area']=data1['Property_Area']
#a=test['Gender']
#b=pd.DataFrame(a)
#encoder=ce.OrdinalEncoder(cols='Gender',mapping=[{'col':'Gender','mapping':{'Male':0,'Female':1}}])
#data2=encoder.fit_transform(b)
##print(data2)
#test['Gender']=data2['Gender']
#a=test['Married']
#b=pd.DataFrame(a)
#encoder=ce.OrdinalEncoder(cols='Married',mapping=[{'col':'Married','mapping':{'Yes':0,'No':1}}])
#data2=encoder.fit_transform(b)
##print(data2)
#test['Married']=data2['Married']
#a=test['Education']
#b=pd.DataFrame(a)
#encoder=ce.OrdinalEncoder(cols='Education',mapping=[{'col':'Education','mapping':{'Graduate':0,'Not Graduate':1}}])
#data2=encoder.fit_transform(b)
##print(data2)
#test['Education']=data2['Education']
#a=test['Self_Employed']
#b=pd.DataFrame(a)
#encoder=ce.OrdinalEncoder(cols='Self_Employed',mapping=[{'col':'Self_Employed','mapping':{'Yes':0,'No':1}}])
#data2=encoder.fit_transform(b)
##print(data2)
#test['Self_Employed']=data2['Self_Employed']
#a=test['Dependents']
#b=pd.DataFrame(a)
#encoder=ce.OrdinalEncoder(cols='Dependents',mapping=[{'col':'Dependents','mapping':{'3+':3}}])
#data2=encoder.fit_transform(b)
##print(data2)
#test['Dependents']=data2['Dependents']
test=test.drop(['Gender','Married','Dependents','Education','Self_Employed','ApplicantIncome','CoapplicantIncome','CoapplicantIncome','Property_Area','Loan_Amount_Term'],axis='columns')
dataf=test
print(dataf)


joblib.dump(dataf,'dataf.xml')
 
 
 
