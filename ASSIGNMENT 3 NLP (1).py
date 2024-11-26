#!/usr/bin/env python
# coding: utf-8

# In[1]:


import spacy
nlp = spacy.load('en_core_web_sm')
with open('peterrabbit.txt', 'r') as f:
    doc = nlp(f.read())


# In[2]:


# Get the third sentence
third_sentence = list(doc.sents)[2]
for token in third_sentence:
    print(f'Text: {token.text:<12} POS: {token.pos_:<6} Tag: {token.tag_:<6} Description: {spacy.explain(token.tag_)}')


# In[3]:


from collections import Counter
# Count POS tags
pos_counts = Counter(token.pos_ for token in doc)
print(pos_counts)


# In[4]:


# Calculate the percentage of nouns
total_tokens = len(doc)
noun_count = sum(1 for token in doc if token.pos_ == 'NOUN')
noun_percentage = (noun_count / total_tokens) * 100

print(f'Percentage of nouns: {noun_percentage:.2f}%')


# In[5]:


from spacy import displacy
# Display dependency parse for the third sentence
displacy.render(third_sentence, style='dep', jupyter=True, options={'distance': 100})


# In[6]:


# Print the first two named entities
for ent in doc.ents[:2]:
    print(f'Entity: {ent.text} - Label: {ent.label_} - Description: {spacy.explain(ent.label_)}')


# In[7]:


# Count sentences
num_sentences = len(list(doc.sents))
print(f'Number of sentences: {num_sentences}')


# In[8]:


# Count sentences with named entities
sentences_with_ners = [sent for sent in doc.sents if sent.ents]
count_sentences_with_ners = len(sentences_with_ners)

print(f'Number of sentences containing named entities: {count_sentences_with_ners}')


# In[9]:


# Display named entity visualization for the first sentence
displacy.render(sentences_with_ners[0], style='ent', jupyter=True)


# In[ ]:




