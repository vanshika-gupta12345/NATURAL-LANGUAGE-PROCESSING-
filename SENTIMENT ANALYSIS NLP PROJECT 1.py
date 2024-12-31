#!/usr/bin/env python
# coding: utf-8

# # Sentiment Analysis (using IMBD dataset)

# In[2]:


import pandas as pd
data= pd.read_csv("IMDB Dataset (1).csv")
data


# In[3]:


data.shape


# In[4]:


data["review"][2]


# In[5]:


#unique values 
data.nunique()


# In[6]:


#slicing 1 to 10000 rows 
df=data.iloc[:10000]
df


# In[7]:


#value counts
df["sentiment"].value_counts()


# In[8]:


#isnull().sum()
df.isnull().sum()


# In[9]:


df.duplicated().sum()


# In[10]:


#dropping duplicates
df.drop_duplicates(inplace=True)


# In[11]:


#check duplicates for verification 
df.duplicated().sum()


# In[12]:


import re
def remove_tags(raw_text):
    cleaned_text=re.sub(re.compile('<.*?>', ' ',raw_text))
    return cleaned_text


# In[14]:


df['review'] = df['review'].apply(remove_tags)


# In[15]:


df


# In[16]:


from nltk.corpus import stopwords
sw_list =stopwords.words('english')


# In[17]:


df['review'] = df['review'].apply(lambda x: [item for item in x.split() if item not in sw_list]).apply(lambda x:" ".join(x))


# In[18]:


df


# In[19]:


X = df.iloc[:,0:1]
y = df['sentiment']
X


# In[20]:


y


# In[27]:


from sklearn.preprocessing import LabelEncoder
encoder= LabelEncoder()
y = encoder.fit_transform(y)


# In[28]:


y


# In[30]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=1)


# In[31]:


X_train.shape


# In[32]:


print(X_train.shape)
print(X_test.shape)
print(y_test.shape)
print(y_train.shape)


# In[33]:


# Applying BOW
from sklearn.feature_extraction.text import CountVectorizer


# In[34]:


cv = CountVectorizer()


# In[35]:


X_train_bow = cv.fit_transform(X_train['review']).toarray()
X_test_bow = cv.transform(X_test['review']).toarray()

# Vimp ) transform only train data ko krte hai.


# In[36]:


X_train_bow.shape


# In[37]:


X_test_bow.shape


# In[38]:


from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train_bow, y_train)


# In[39]:


y_pred = gnb.predict(X_test_bow)


# In[40]:


from sklearn.metrics import accuracy_score, confusion_matrix
accuracy_score(y_test, y_pred)


# In[41]:


confusion_matrix(y_test,y_pred)


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
rf.fit(X_train_bow, y_train)
y_pred = rf.predict(X_test_bow)


# In[ ]:


accuracy_score(y_test, y_pred)


# In[ ]:


cv = CountVectorizer(max_features=3000)


# In[ ]:


X_train_bow = cv.fit_transform(X_train['review']).toarray()
X_test_bow = cv.transform(X_test['review']).toarray()
rf = RandomForestClassifier()
rf.fit(X_train_bow, y_train)
y_pred = rf.predict(X_test_bow)


# In[ ]:


accuracy_score(y_test, y_pred)


# In[ ]:


cv = CountVectorizer(ngram_range=(1,2),max_features=5000)
X_train_bow = cv.fit_transform(X_train['review']).toarray()
X_test_bow = cv.transform(X_test['review']).toarray()
rf = RandomForestClassifier()
rf.fit(X_train_bow, y_train)
y_pred = rf.predict(X_test_bow)
accuracy_score(y_test, y_pred)


# In[ ]:




