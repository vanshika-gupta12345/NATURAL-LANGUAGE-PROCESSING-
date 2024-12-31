#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importing the libraries and dataset
import spacy
from nltk.corpus import stopwords
from string import punctuation


# In[2]:


nlp=spacy.load('en_core_web_sm')


# In[3]:


text = """ Maria Sharapova has basically no friends as tennis players on the WTA Tour. The Russian player has no problems in openly speaking about it and in a recent interview she said: ‘I don’t really hide any feelings too much.
I think everyone knows this is my job here. When I’m on the courts or when I’m on the court playing, I’m a competitor and I want to beat every single person whether they’re in the locker room or across the net.
So I’m not the one to strike up a conversation about the weather and know that in the next few minutes I have to go and try to win a tennis match.
I’m a pretty competitive girl. I say my hellos, but I’m not sending any players flowers as well. Uhm, I’m not really friendly or close to many players.
I have not a lot of friends away from the courts.’ When she said she is not really close to a lot of players, is that something strategic that she is doing? Is it different on the men’s tour than the women’s tour? ‘No, not at all.
I think just because you’re in the same sport doesn’t mean that you have to be friends with everyone just because you’re categorized, you’re a tennis player, so you’re going to get along with tennis players.
I think every person has different interests. I have friends that have completely different jobs and interests, and I’ve met them in very different parts of my life.
I think everyone just thinks because we’re tennis players we should be the greatest of friends. But ultimately tennis is just a very small part of what we do.
There are so many other things that we’re interested in, that we do. """


# In[6]:


sw=set(stopwords.words('english'))


# In[7]:


doc = nlp(text) # applied tokenization


# In[8]:


tokens=[token.text for token in doc]
print(tokens)


# In[9]:


# add new line (\n) to punctuation
punctuation = punctuation + '\n'


# In[10]:


punctuation


# # Text Cleaning

# In[13]:


word_frequencies = {}

for word in doc:
    if word.text.lower() not in sw:
        if word.text.lower() not in punctuation:
            if word.text not in word_frequencies.keys():
                word_frequencies[word.text] = 1
            else:
                word_frequencies[word.text] +=1


# In[15]:


print(word_frequencies)


# In[16]:


max_frequencies = max(word_frequencies.values())


# In[17]:


max_frequencies


# In[18]:


for word in word_frequencies.keys():
    word_frequencies[word] = word_frequencies[word] / max_frequencies


# In[19]:


print(word_frequencies)


# In[26]:


sentence_tokens = [sent for sent in doc.sents]
print(sentence_tokens)


# In[28]:


sentence_score = {}

for sent in sentence_tokens:
    for word in sent:
        if word.text.lower() in word_frequencies.keys():
            if sent not in sentence_score.keys():
                sentence_score[sent] = word_frequencies[word.text.lower()]
            else:
                sentence_score[sent] += word_frequencies[word.text.lower()]


# In[29]:


print(sentence_score)


# In[30]:


18*(30/100)


# In[31]:


from heapq import nlargest


# In[32]:


select_length = int(len(sentence_tokens)) * 0.3


# In[33]:


print(select_length)
# 30% of total sentences is almost 8


# In[34]:


summary = nlargest(n=int(select_length), iterable=sentence_score, key = sentence_score.get)


# In[35]:


print(summary)
# these 8 sentence represents summary of text


# In[36]:


# combine these sentence together
final_summary = [word.text for word in summary]
final_summary


# In[37]:


len(text) # length of original text


# In[38]:


# the length of summary is almost 30% original length of text


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




