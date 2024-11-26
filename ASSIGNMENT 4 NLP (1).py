#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install spacy')
import spacy
nlp = spacy.load('en_core_web_sm')
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize, RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, LancasterStemmer, WordNetLemmatizer
import string

# Download required resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

# Initialize objects
stop_words = set(stopwords.words("english"))
porter = PorterStemmer()
lancaster = LancasterStemmer()
lemmatizer = WordNetLemmatizer()
regex_tokenizer = RegexpTokenizer(r'\w+')


# In[2]:


# question 1:Tokenize a simple sentence using word_tokenize. ( "Natural Language Processing with Python is fun")
doc = nlp(u'natural langugae processing is fun with pyhton ')
for token in doc:
    print(token.text)


# In[3]:


# Question 2: Remove punctuation from a sentence
sentence= "Hello there! How's the weather today?"
tokens= word_tokenize(sentence)
punct= [word for word in tokens if word.isalnum()]
print(punct)


# In[4]:


# Question 3: Remove stopwords from a sentence
sentence = "This is a simple sentence for stopword removal."
tokens= word_tokenize(sentence)
filtered= [word for word in tokens if word.lower() not in stop_words]
print( filtered)


# In[5]:


# Question 4: Perform stemming using PorterStemmer
sentence= "The striped bats are hanging on their feet for best."
tokens= word_tokenize(sentence)
stem= [porter.stem(word) for word in tokens]
print(stem)


# In[6]:


# Question 5: Perform lemmatization using WordNetLemmatizer
sentence= "The geese are flying south for the winter."
tokens= word_tokenize(sentence)
lemmatized= [lemmatizer.lemmatize(word) for word in tokens]
print(lemmatized)


# In[7]:


# Question 6: Convert text to lowercase and remove punctuation
sentence = "Hello, World! NLP with Python."
lower = sentence.lower()
punct = ''.join([char for char in lower if char not in string.punctuation])
print(punct)


# In[8]:


# Question 7: Tokenize a sentence into sentences
sentence = "Hello World. This is NLTK. Let's explore NLP!"
sentences = sent_tokenize(sentence)
print(sentences)


# In[9]:


# Question 8: Stem words in a sentence using LancasterStemmer
sentence = "Loving the experience of learning NLTK"
tokens = word_tokenize(sentence)
stem = [lancaster.stem(word) for word in tokens]
print(stem)


# In[10]:


# Question 9: Remove both stopwords and punctuation from a sentence
sentence = "This is a test sentence, with stopwords and punctuation!"
tokens= word_tokenize(sentence)
filtered = [word for word in tokens if word.lower() not in stop_words and word.isalnum()]
print(filtered)


# In[11]:


# Question 10: Lemmatize words with their part-of-speech (POS) tag
from nltk.corpus import wordnet

def get_wordnet_pos(word):
    """Map POS tag to first character for lemmatizer."""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ, "N": wordnet.NOUN, "V": wordnet.VERB, "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)

sentence = "The striped bats are hanging on their feet."
tokens = word_tokenize(sentence)
lemmatized = [lemmatizer.lemmatize(word, get_wordnet_pos(word)) for word in tokens]
print(lemmatized)


# In[12]:


# Question 11: Tokenize and remove stopwords, punctuation, and perform stemming
sentence = "Running through the forest, the fox is faster."
tokens = word_tokenize(sentence)
filtered = [porter.stem(word) for word in tokens if word.lower() not in stop_words and word.isalnum()]
print(filtered)


# In[13]:


# Question 12: Count stopwords in a sentence
sentence = "This is an example sentence for counting stopwords."
tokens = word_tokenize(sentence)
stopword_count = len([word for word in tokens if word.lower() in stop_words])
print(stopword_count)


# In[14]:


# Question 13: Perform stemming and remove punctuation using RegexTokenizer
sentence = "Stemming, punctuation! Removal example."
tokens = regex_tokenizer.tokenize(sentence)
stem = [porter.stem(word) for word in tokens]
print(stem)


# In[15]:


# Question14: Remove punctuation using regex and NLTK
sentence = "Punctuation removal with regex in NLP!"
tokens = regex_tokenizer.tokenize(sentence)
print(tokens)


# In[16]:


# Question 15: Tokenize text into words, remove stopwords, and lemmatize
sentence = "The dogs are barking loudly."
tokens = word_tokenize(sentence)
filter = [lemmatizer.lemmatize(word) for word in tokens if word.lower() not in stop_words and word.isalnum()]
print(filter)


# In[ ]:





# In[ ]:




