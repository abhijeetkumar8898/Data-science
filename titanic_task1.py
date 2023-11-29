#!/usr/bin/env python
# coding: utf-8

# In[50]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import seaborn as sn


# In[5]:


test = pd.read_csv('tested.csv')


# In[7]:


test.head()


# In[8]:


test.columns


# In[14]:


test.dtypes
test.info()


# In[11]:


#univariate analysis
test['Survived'].value_counts()


# In[12]:


test['Survived'].value_counts(normalize=True)


# In[13]:


test['Survived'].value_counts().plot.bar()


# In[15]:


test['Sex'].value_counts()


# In[16]:


test['Sex'].value_counts().plot.bar()


# In[27]:


sns.countplot(x='Sex',hue='Survived',data=test).set_title('Passeneger survived after collision')


# In[32]:


sn.countplot(x='Pclass',hue='Survived',data=test).set_title('Passeneger survived in different class for males and female')


# In[33]:


test.head()


# In[39]:


p=test.drop(columns = ['PassengerId','Name','Ticket','Survived'],axis = 1)
q= test['Survived']
print(p)


# In[40]:


print(q)


# In[43]:


test.isnull().sum()


# In[44]:


test=test.drop(columns ='Cabin',axis =1)


# In[45]:


test.head()


# In[46]:


test['Age'].fillna(test['Age'].mean(),inplace = True)


# In[47]:


test.isnull().sum()


# In[48]:


test['Fare'].fillna(test['Fare'].mean(),inplace = True)


# In[49]:


test.isnull().sum()


# In[62]:


x = test.drop(['Survived'],axis=1)
y=test['Survived']


# In[64]:


from sklearn .model_selection import train_test_split


# In[65]:


train_x,test_x,train_y,test_y=train_test_split(x,y,random_state=2,stratify=y)


# In[66]:


train_y.value_counts()/len(train_y)


# In[67]:


test_y.value_counts()/len(test_y)


# In[77]:


#importing decision tree
from sklearn.tree import DecisionTreeClassifier


# In[78]:


clf = DecisionTreeClassifier()
clf.fit(train_x,train_y)
clf.score(train_x,train_y)


# In[81]:


from sklearn import tree
plt.figure(figsize = (20,15))
tree.plot_tree(clf,filled = True)


# In[80]:


clf.score(test_x,test_y)
clf.predict(test_x)


# In[ ]:




