#!/usr/bin/env python
# coding: utf-8

# In[42]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import seaborn as sn
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler


# In[43]:


data =pd.read_csv('iris.csv')


# In[44]:


data.head()



# In[45]:


data.describe


# In[46]:


data.shape


# In[60]:


data.isnull().sum()


# In[61]:


out=data.drop(['petal_length','petal_width','species'],axis=1)
out


# In[62]:


kmeans = KMeans(n_clusters=4)


# In[63]:


y_predicted=kmeans.fit_predict(out[['sepal_length','sepal_width']])
y_predicted


# In[64]:


out['cluster']=y_predicted
out


# In[65]:


out.head(100)


# In[66]:


out1=out[out.cluster==0]
out2=out[out.cluster==1]
out3=out[out.cluster==2]


# In[67]:


plt.scatter(out1.sepal_length,out1.sepal_width,color='red')
plt.scatter(out2.sepal_length,out2.sepal_width,color='blue')
plt.scatter(out3.sepal_length,out3.sepal_width,color='violet')


# In[ ]:





# In[ ]:




