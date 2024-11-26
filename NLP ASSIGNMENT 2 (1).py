#!/usr/bin/env python
# coding: utf-8

# In[2]:


get_ipython().system('pip install spacy')


# In[3]:


import spacy 
nlp=spacy.load('en_core_web_sm')


# # 1. Create a Doc object from the file owlcreek.txt

# In[4]:


with open ('owlcreek.txt','r') as a:
    fields = a.read()
fields


# # 2. How many tokens are contained in the file?

# In[11]:


doc = nlp(u'A man stood upon a railroad bridge in northern Alabama, looking downinto the swift water twenty feet below.')

for token in doc:
    print(token.text)


# # 3. How many sentences are contained in the file?

# In[6]:


len(doc)


# # 4. Print the second sentence in the document
# 
# 

# In[8]:


doc[1]


# # 5. For each token in the sentence above, print its text, POS tag, dep tag and lemma.
# 
# 

# In[9]:


for token in doc:
    print(token.text,token.pos_,token.dep_,token.lemma_)


# # 6. Write a matcher called 'Swimming' that finds both occurrences of the phrase "swimming vigorously" in the text.
# 
# 

# In[ ]:


import spacy 
nlp = spacy.load('en_core_web_sm')

from spacy.matcher import PhraseMatcher 

# create a PhraseMatcher object
phrase_matcher = PhraseMatcher(nlp.vocab)

# Define phrases to match
phrases = ["swimming\vigorously"]
patterns = [nlp(text) for text in phrases]
phrase_matcher.add("Pattern",patterns)

doc = nlp(doc)
matches = phrase_matcher(doc)
matches


# In[ ]:


# Display matches
for match_id, start, end in found_matcher:
    # Print the match and its surrounding context
    print(f"Match: {doc2[start:end].text}")
    print(f"Surrounding text: {doc2[max(0, start-5):min(len(doc2), end+5)].text}")
    print("-" * 40)


# In[ ]:




