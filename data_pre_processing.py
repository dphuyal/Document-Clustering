#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 24 19:09:21 2018

@author: phuyaldeep

Sources cited:
    https://stackoverflow.com/questions/14798220/how-can-i-search-sub-folders-using-glob-glob-module-in-python
    https://stackoverflow.com/questions/19699367/unicodedecodeerror-utf-8-codec-cant-decode-byte
    https://machinelearningmastery.com/clean-text-machine-learning-python/
    https://machinelearningmastery.com/develop-word-embeddings-python-gensim/

"""
import glob
import string
import time
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from gensim.models import Word2Vec
import numpy as np
import pandas as pd
#import math

t=time.time()

# returns a list of all the sorted text files
total_files = sorted(glob.glob('Part1/*/*/*.txt', recursive=True))

# to store all the text after 'Abstract'
temp_data = []

# loops through the length of all the files and extracts all the sentences after 'Abstract'
for f in range(len(total_files)):
    lines_after_abstract = ""
    with open(total_files[f],encoding='iso-8859-1') as read_files:
        read_files = read_files.read()
        read_files = read_files.split()
        pointer = read_files.index('Abstract')
        #print (pointer)
        for lines in read_files[pointer + 2:]:
            lines_after_abstract = lines_after_abstract + lines + " "
    if 'Not Available' in lines_after_abstract:
        #print('Not Available found')
        continue
    else:
        temp_data.append(lines_after_abstract)

stop_words = set(stopwords.words('english'))
porter = PorterStemmer()
table = str.maketrans('', '', string.punctuation)

# stores all the doc after tokenizing, stemming and removing punctuation, and removing stop words
doc_list = []

for temp in temp_data:
    tokens = word_tokenize(temp)
    stripped = [w.translate(table) for w in tokens]
    stemmed = [porter.stem(word) for word in stripped]
    words = [word for word in stemmed if word.isalpha()]
    words = [w for w in words if not w in stop_words]
    doc_list.append(words)

#print("Done in  %0.3fs." % (time.time() - t))

# train model
model = Word2Vec(doc_list, size=300, window=5, workers=4, min_count=1)
# vocabularies. This is the actual word vector model
word_vecs = model.wv

#print("Done in  %0.3fs." % (time.time() - t))

df = pd.DataFrame(columns = list(range(300)))
i = 0
for doc in doc_list:
    # ingores the words that are not present in the vocabulary
    doc2Vec = []
    for word in doc:
        if word in word_vecs:
            w2v = model[word]
            doc2Vec.append(w2v)
    average = np.mean(doc2Vec, axis = 0)
    df.loc[i] = average
    i += 1

#print('Total words shape: ', np.shape(total_words))
#print('first document size: ', np.shape(total_words[0]))
#print(total_words[:5])

df.to_csv('doc2vec.csv')

print("Done in  %0.3fs." % (time.time() - t))
