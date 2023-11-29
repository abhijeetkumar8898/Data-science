#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import seaborn as sn


# In[17]:


data =pd.read_csv('advertising.csv')


# In[18]:


data.head()


# In[19]:


data.dtypes
data.info()


# In[20]:


data['Sales'].value_counts()


# In[21]:


data['Sales'].value_counts(normalize=True)


# In[22]:


data.describe()


# In[23]:


data.shape


# In[24]:


data.isnull()


# In[25]:


data.sum()


# In[27]:


data.isnull().sum()


# In[30]:


import seaborn as sns
fig,axs=plt.subplots(3,figsize =(8,8))
plt1 = sns.boxplot(data['TV'],ax = axs[0])
plt2 = sns.boxplot(data['Newspaper'],ax = axs[1])
plt3 = sns.boxplot(data['Radio'],ax = axs[2])
plt.tight_layout()


# In[39]:


sns.pairplot(data,x_vars=['TV','Radio','Newspaper'],y_vars='Sales',kind='scatter')
plt.show()


# In[54]:


plt.hist(x='TV',data=data,bins=10)
plt.show()


# In[51]:


plt.hist(x='Radio',data=data,color="red",bins=10)
plt.show()


# In[52]:


plt.hist(x='Newspaper',data=data,color="orange",bins=10)
plt.show()


# In[56]:


sns.heatmap(data.corr(),annot=True)
plt.show
#basically we can see that sales is mostly releated with the advertisement of tv we can conclude sales is related with tv


# # linear regression

# In[78]:


from sklearn.linear_model import LinearRegression
lreg=LinearRegression()
from sklearn .model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(data[['TV']],data[['Sales']],test_size=0.4,random_state=0)


# In[71]:


print(x_train)


# In[73]:


print(y_train)


# In[74]:


lreg=LinearRegression()


# In[88]:


lreg.fit(x_train,y_train)


# In[85]:


print(y_test)


# In[89]:


result=lreg. predict(x_test)
print(result)


# In[ ]:


#our prediction and y_train value is almost accurate so we caan conclude that we have predicted correct


# In[91]:


lreg.coef_


# In[97]:


lreg.intercept_


# In[98]:


0.05565446*69.2+7.02141391


# In[99]:


plt.scatter(x_test,y_test)
plt.plot(x_test,7.02141391+0.05565446*x_test,'g')
plt.show()


# In[ ]:




