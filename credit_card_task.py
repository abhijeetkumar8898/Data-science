#!/usr/bin/env python
# coding: utf-8

# In[20]:


import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
pd.set_option('display.max_columns',None)
import seaborn as sn


# In[4]:


data =pd.read_csv('creditcard.csv')


# In[5]:


data.head()


# In[7]:


data.shape


# In[8]:


data.dtypes


# In[9]:


data.isnull().sum()


# In[26]:


#checking. balance
fraud=data["Class"].value_counts()
fraud_rate=100*(fraud/data.shape[0])
fraud_data=pd.concat([fraud,fraud_rate],axis=1).reset_index()
fraud.columns=['class','count','Percentage']
fraud 
fraud_data


# In[30]:


#imbalance
data_fraud=data[data['Class']==1]
data_not_fraud=data[data['Class']==0]
data_not_fraud_sampled=data_not_fraud.sample(data_fraud.shape[0],replace=False,random_state=101)
data_balanced=pd.concat([data_not_fraud_sampled,data_fraud],axis=0).sample(frac=1,replace=False,random_state=101).reset_index().drop('index',axis=1)
data_balanced


# In[33]:


#checking. balance
fraud=data_balanced["Class"].value_counts()
fraud_rate=100*(fraud/data_balanced.shape[0])
fraud_data=pd.concat([fraud,fraud_rate],axis=1).reset_index()
fraud.columns=['class','count','Percentage']
fraud 
fraud_data


# In[35]:


#train test split
x_train,x_test,y_train,y_test=train_test_split(data_balanced.drop('Class',axis=1),data_balanced['Class'],test_size=0.2,random_state=101)


# In[38]:


print("x_train:",{x_train.shape})
print("x_test:",{x_test.shape})
print("y_train:",{y_train.shape})
print("y_test:",{y_test.shape})


# # Random forest
# 
# 

# In[41]:


#model pipe
randomForestModel = Pipeline([
    ('scaler', StandardScaler()),  # Note: 'Scaler' changed to 'scaler'
    ('classifier', RandomForestClassifier())  # Note: 'RandomForestClassiifier' changed to 'RandomForestClassifier'
])


# In[45]:


randomForestModel.fit(x_train, y_train)


# In[46]:


y_pred = randomForestModel.predict(x_test)
y_pred


# In[47]:


cla=classification_report(y_test,y_pred)


# In[48]:


print(cla)


# In[52]:


with open('./model.pk1','wb')as fp:
    pickle.dump(randomForestModel,fp)


# In[ ]:




