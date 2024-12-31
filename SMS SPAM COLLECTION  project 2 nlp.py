#!/usr/bin/env python
# coding: utf-8

# In[22]:


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt


# In[3]:


df=pd.read_csv("smsspamcollection.tsv",sep="\t")


# In[4]:


df


# In[5]:


df.head()


# In[41]:


df.shape


# In[6]:


df.tail()


# In[7]:


df.isnull().sum()


# In[8]:


df.nunique()


# In[9]:


df.describe()


# In[12]:


df['label'].value_counts()


# # Balancing the data

# In[13]:


# select ham data
ham = df[df['label']=='ham']
ham.head()

# taking all ham data in new ham variable.


# In[14]:


spam = df[df['label']=='spam']
spam.head()


# In[15]:


ham.shape, spam.shape


# In[16]:


spam.shape[0]


# In[17]:


ham=ham.sample(spam.shape[0])


# In[18]:


# check the shape of data
# size off ham and spam data is same, now this is the balanced data
ham.shape, spam.shape


# In[19]:


data = pd.concat([ham,spam],ignore_index=True)


# In[20]:


data.shape


# In[21]:


data


# # Data Visualization

# In[23]:


# plot histogram of length for ham messages
plt.hist(data[data['label'] == 'ham']['length'], bins=100, alpha=0.7)
plt.show()
# from the histogram we can say that, the number of characters in ham messages are less than 1)


# In[24]:


plt.hist(data[data['label']=='ham']['length'],bins=100,alpha=0.7)
plt.hist(data[data['label']=='spam']['length'],bins=100,alpha=0.7)
plt.show()


# In[25]:


plt.hist(data[data['label']=='ham']['punct'],bins=100,alpha=0.7)
plt.hist(data[data['label']=='spam']['punct'],bins=100,alpha=0.7)
plt.show()


# In[29]:


from sklearn.model_selection import train_test_split


# In[30]:


x_train, x_test, y_train, y_test = train_test_split(data['message'],data['label'],test_size=0.3,
                                                  
                                                                   random_state=0, shuffle=True)


# In[ ]:





# In[26]:


from sklearn.pipeline import Pipeline
# there will be lot of repeated processes for training and testing the dataset separately,
# to avoid that we are using pipeline

from sklearn.feature_extraction.text import TfidfVectorizer
# we are importing TfidfVectorizer to utilize bag of words model in sklearn

from sklearn.ensemble import RandomForestClassifier


# In[27]:


classifier = Pipeline([('tfidf', TfidfVectorizer()), ('classifier',RandomForestClassifier(n_estimators=100))])


# In[31]:


classifier.fit(x_train, y_train)


# In[32]:


from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# In[33]:


y_pred = classifier.predict(x_test)


# In[34]:


# confusion_matrix
confusion_matrix(y_test, y_pred)


# In[35]:


# classification report
print(classification_report(y_test,y_pred))

# we are getting almost 95% accuracy


# In[36]:


accuracy_score(y_test, y_pred)


# In[37]:


# Predict a real message
classifier.predict(['Hello, You are learning natural language Processing'])


# In[38]:


classifier.predict(['Hope you are doing good and learning new things !'])


# In[39]:


classifier.predict(['Congraturation, you won 50 crore'])


# In[ ]:




