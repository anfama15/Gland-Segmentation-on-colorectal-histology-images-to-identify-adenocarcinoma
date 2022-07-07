# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data=pd.read_csv('dataset1.csv')
x=data.iloc[:,:-1]
y=data.iloc[:,-1]

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder
labelencoder=LabelEncoder()
# y=labelencoder.fit_transform(y)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)


from sklearn.neighbors import KNeighborsClassifier 
classifier1=KNeighborsClassifier(n_neighbors=5, metric='minkowski',p=2) 
classifier1.fit(x_train,y_train) 
y_pred1=classifier1.predict(x_test) 
print('KNN results:', y_pred1) 
from sklearn.metrics import confusion_matrix, accuracy_score 
CM1=confusion_matrix(y_test,y_pred1) 
print('KNN Confusion Matrix: ',CM1) 
S_KNN=accuracy_score(y_test,y_pred1)
print('KNN Accuracy:',S_KNN)

from sklearn.naive_bayes import GaussianNB
Classifier2=GaussianNB()
Classifier2.fit(x_train,y_train) 
y_pred11=Classifier2.predict(x_test) 
print('Naive Bayes Results:', y_pred11) 
CM2=confusion_matrix(y_test,y_pred11) 
print('Naive Bayes Confusion Matrix: ',CM2) 
S_NB=accuracy_score(y_test,y_pred11)
print('Naive Bayes Accuracy:',S_NB)


from sklearn.tree import DecisionTreeClassifier
Classifier3=DecisionTreeClassifier(criterion='entropy',random_state=0)
Classifier3.fit(x_train,y_train) 
y_pred111=Classifier3.predict(x_test) 
print('Decision Tree Classifier Results:', y_pred111) 
CM3=confusion_matrix(y_test,y_pred111) 
print('Decision Tree Classifier Confusion Matrix:',CM3) 
S_DTC=accuracy_score(y_test,y_pred111)
print('Decision Tree Classifier Accuracy',S_DTC)


from sklearn.ensemble import RandomForestClassifier
clf=RandomForestClassifier(n_estimators=10)
clf.fit(x_train,y_train)
y_pred1111=clf.predict(x_test)
print('Random Forest Classifier Results:', y_pred1111) 
CM4=confusion_matrix(y_test,y_pred1111) 
print('Random Forest Classifier Confusion Matrix:',CM4) 
S_RFC=accuracy_score(y_test,y_pred1111)
print('Random Forest Classifier Accuracy',S_RFC)
