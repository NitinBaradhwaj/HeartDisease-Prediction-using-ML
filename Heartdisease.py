#!/usr/bin/env python
# coding: utf-8

# In[1]:


#We follow four steps
#Data Loading --> Data Preprocessing and Visualization --> Train Test Split --> Logistic Regression Model
#Import Dependencies


# In[98]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


# In[61]:


#Load the dataset 
heartds=pd.read_csv('~/Desktop/Datasets/heart.csv')


# In[62]:


#complete dataset is loaded
heartds


# In[63]:


# target is 0 ---> Healthy Heart, safe no heart disease
# target is 1 ---> Defective Heart, means you heart disease 


# In[64]:


#printing first five rows
heartds.head()


# In[65]:


#let's check the last five rows also
heartds.tail()


# In[66]:


#total rows and columns
heartds.shape


# In[67]:


heartds.info()


# In[68]:


heartds.groupby('target').size()


# In[69]:


heartds['target'].value_counts()
#the above step and this step both are used to check a column's distribution


# In[70]:


heartds['age'].value_counts()


# In[71]:


#total size of the dataset
heartds.size


# In[72]:


#statistics of our dataset
heartds.describe()


# In[73]:


#check for null values
heartds.isnull().sum()


# In[74]:


#There are no null values in our dataset


# In[75]:


#Data Visualization


# In[127]:


#Train_Test_split
#into X_train,X_test,Y_train,Y_train
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# In[128]:


sc = StandardScaler()
X = sc.fit_transform(X)


# In[129]:


Y


# In[130]:


X


# In[131]:


#Splitting into train and test data
X_train , X_test , Y_train , Y_test = train_test_split(X,Y,random_state=10,test_size=0.3,shuffle=True)
#Using random state data is split randomly


# In[132]:


X_train


# In[133]:


Y_train


# In[134]:


X_test


# In[135]:


Y_test


# In[136]:


#Checking the shape of train and test data
X.shape, X_train.shape, X_test.shape


# In[137]:


Y.shape, Y_train.shape, Y_test.shape


# In[138]:


scores_dict={}


# In[200]:


Catagory=['No....but i pray you get Heart Disease or at leaset Corona Virus Soon...','Yes you have Heart Disease....RIP in Advance']


# In[139]:


#MODEL TRAINING
#USING LOGISTIC REGRESSION


# In[147]:


#we are going to use logistic regression
mymodel = DecisionTreeClassifier()


# In[148]:


#training our model with train data


# In[149]:


mymodel.fit(X_train, Y_train)


# In[150]:


#evaluationg our model 


# In[151]:


X_train


# In[154]:


predicting_output = mymodel.predict(X_test)
accuracy_fit=accuracy_score(Y_test,predicting_output)*100


# In[158]:


scores_dict['DecisionTreeClassifier'] = accuracy_fit
print('Accuracy : ',accuracy_fit)


# In[160]:


print("Accuracy on training set: {:.3f}".format(mymodel.score(X_train, Y_train)))


# In[161]:


print("Accuracy on test set: {:.3f}".format(mymodel.score(X_test, Y_test)))


# In[162]:


predicting_output


# In[201]:


X_DT=np.array([[63 ,1, 3,145,233,1,0,150,0,2.3,0,0,1]])
X_DT_predicting_output=mymodel.predict(X_DT)


# In[211]:


X_DT_predicting_output[0]

print(Catagory[int(X_DT_predicting_output[0])])

