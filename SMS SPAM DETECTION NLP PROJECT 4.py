#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


df = pd.read_csv("smsspamcollection.tsv",sep="\t")
df.head()


# In[3]:


df.sample(5) # randomly 5 rows


# In[4]:


df.shape


# In[5]:


# 1. Data Cleaning
# 2. EDA
# 3. Text Preprocessing
# 4. Model Building
# 5. Evaluation
# 6. Improvement depending on evaluation
# 7. website


# # Data Cleaning

# In[6]:


df.isnull().sum()


# In[7]:


df.info()


# In[8]:


df1=df.drop(columns=['length','punct'])


# In[9]:


df1.sample(5)


# In[11]:


#label encoder
from sklearn.preprocessing import LabelEncoder
encoder=LabelEncoder()


# In[12]:


encoder.fit_transform(df1['label'])


# In[13]:


df1['label']=encoder.fit_transform(df1['label'])


# In[14]:


df1.head()


# In[15]:


# checking duplicates and 
df1.isnull().sum()


# In[16]:


df1.duplicated().sum()


# In[17]:


# removing duplicates
df1.drop_duplicates(keep='first',inplace=True)
df1


# In[18]:


df1.duplicated().sum()


# In[19]:


df1.shape


# # EDA

# In[20]:


df1.head()


# In[21]:


df1['label'].value_counts()


# In[22]:


import matplotlib.pyplot as plt
plt.pie(df1['label'].value_counts(), labels=['ham','spam'], autopct = "%0.2f")
plt.show()
# autopct parameter in pie() allows you to format the percentage labels that appear on the pie


# In[24]:


get_ipython().system('pip install nltk')


# In[25]:


import nltk


# In[26]:


nltk.download('punkt')


# In[27]:


df1['message'].apply(len)


# In[28]:


df1['num_characters']=df['message'].apply(len)


# In[29]:


df1


# In[30]:


# Number of words
df1['message'].apply(lambda x:len(nltk.word_tokenize(x)))


# In[31]:


df1['num_words'] = df1['message'].apply(lambda x:len(nltk.word_tokenize(x)))


# In[32]:


df1.head()


# In[33]:


df1['message'].apply(lambda x:(nltk.sent_tokenize(x)))


# In[34]:


df1['num_sentences'] = df1['message'].apply(lambda x:len(nltk.sent_tokenize(x)))


# In[35]:


df1[['num_words','num_characters','num_sentences']].describe()


# In[36]:


# ham
df1[df1['label']==0][['num_words','num_characters','num_sentences']].describe()


# In[37]:


# spam
df1[df1['label']==1][['num_words','num_characters','num_sentences']].describe()


# In[38]:


import seaborn as sns
import matplotlib.pyplot as plt


# In[39]:


plt.figure(figsize=(12,6))
sns.histplot(df1[df1['label'] == 0]['num_characters'])
sns.histplot(df1[df1['label'] == 1]['num_characters'],color = 'red')


# In[40]:


plt.figure(figsize=(12,6))
sns.histplot(df1[df1['label'] == 0]['num_sentences'])
sns.histplot(df1[df1['label'] == 1]['num_sentences'],color = 'red')


# In[41]:


plt.figure(figsize=(12,6))
sns.histplot(df1[df1['label'] == 0]['num_words'])
sns.histplot(df1[df1['label'] == 1]['num_words'],color = 'red')

# spam start from 0 
# ham start from approx 25
# so not the big different
# no. of ham and spam ke base pe hm ni bta pa rhe hai ki spam ke words jada hai ya spam ke jada hai.


# In[42]:


sns.pairplot(df1,hue='label')


# In[43]:


df2 = df1.drop(columns=['message'])
df2


# In[44]:


df2.corr()


# In[45]:


plt.figure(figsize=(5,5))
sns.heatmap(df2.corr(),annot=True,square=True)


# In[46]:


# 1. Lower case
# 2. Tokenization
# 3. Remove the characters
# 4. Removing stop words and punctuations
# 5. Stemming 


# In[47]:


from nltk.corpus import stopwords


# In[48]:


import string
string.punctuation


# In[49]:


from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()


# In[50]:


def transform_text(text):
    text = text.lower()
    
    # Tokenize text
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:] 
    y.clear()   

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()   

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)


# In[51]:


nltk.download('stopwords')


# In[52]:


df1['message'].apply(transform_text)


# In[53]:


df1.head()


# In[60]:


df1['transformed_message']=df1['message'].apply(transform_text)


# In[61]:


df1.head()


# In[54]:


get_ipython().system('pip install wordcloud')


# In[55]:


from wordcloud import WordCloud


# In[56]:


wc=WordCloud(width=500,height=500,min_font_size=10,background_color='white')


# In[62]:


spam_wc=wc.generate(df1[df1['label']==1]['transformed_message'].str.cat(sep=" "))


# In[63]:


plt.imshow(spam_wc)


# In[64]:


df1[df1['label']==1]['transformed_message'].tolist()


# In[65]:


spam_corpus = []
for msg in df1[df1['label']==1]['transformed_message'].tolist():
    for word in msg.split():
        spam_corpus.append(word)


# In[66]:


len(spam_corpus)


# In[67]:


from collections import Counter
Counter(spam_corpus)


# In[68]:


# Adding in Dataframe
from collections import Counter
df3 = pd.DataFrame(Counter(spam_corpus).most_common(30))
df3


# In[69]:


df3 = df3.rename(columns={0:'Word',1:'Count'})
df3


# In[70]:


import seaborn as sns
import matplotlib.pyplot as plt
sns.barplot(x='Word',y='Count',data = df3)
plt.xticks(rotation='vertical')
plt.show()


# In[ ]:




