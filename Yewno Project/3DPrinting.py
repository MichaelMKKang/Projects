# -*- coding: utf-8 -*-
"""
Created on Tue May  8 13:09:29 2018

@author: Michael Kang
"""

'''
from tqdm import *


#https://newsapi.org/v2/everything?q=3D%20print&from=2018-04-01%to=2018-05-08&language=en&pageSize=100&apiKey=58fb581ea2474038b5e3aa5e9377f172
newsapi = NewsApiClient(api_key='58fb581ea2474038b5e3aa5e9377f172')

all_articles2 = newsapi.get_everything(q='3D print',
                                      #from_parameter='2018-04-01',
                                      #to='2018-05-08',
                                      language='en',
                                      sort_by='popularity',
                                      page=2)

dict_list = []
current_page=1
for i in tqdm(range(10000)):    
    dict_list.append(newsapi.get_everything(q='3D print',
                                      #from_parameter='2018-04-01',
                                      #to='2018-05-08',
                                      language='en',
                                      sort_by='popularity',
                                      page=current_page)['articles'])
'''

#using these to quickly get 
import requests
url=('https://newsapi.org/v2/everything?q=3D%20print&language=en&pageSize=100$page=1&apiKey=58fb581ea2474038b5e3aa5e9377f172')
response = requests.get(url).json()




from newsapi import NewsApiClient
import json
import pandas as pd
import re
import nltk
from collections import Counter
from nltk.corpus import stopwords
import string
punctuation = list(string.punctuation)
stop = stopwords.words('english') + punctuation + ['RT', 'via', '…']




    
def tokenize(s):
    return tokens_re.findall(s)
 
def preprocess(s, lowercase=False):
    tokens = s.split()
    if lowercase:
        tokens = [token if emoticon_re.search(token) else token.lower() for token in tokens]
    return tokens
    
def get_text(path):
    news_text = []
    text_itself = []
    with open(path,'r') as f:
        news_file = json.load(f)['articles']
    for line in news_file:
        try:
            terms_only = [term for term in preprocess(line['description']) if term not in stop and not term.startswith(('#', '@', 'http'))]
            news_text.append(terms_only)
            text_itself.append(line['description'])
        except:
            continue
    #print(count_all.most_common(5))
    return news_text,text_itself

#Can use both the tokenized versions or just the descriptions themselves.
text,text_list = get_text('C:\\Users\\Michael Kang\\Desktop\\3dprinting1.json')             # sample obtained using NewsAPI


#Most common words
word_list=[]
for blurbs in text:
    word_list.extend(blurbs)
words = [w.lower() for w in word_list if (len(w)>=2 and not 'print' in w)]               #getting rid of single character words and making all lowercase
words = nltk.FreqDist(words)
mostcommon = words.most_common(50)                     #arbitrary threshold for now... Also still noticing 'print' variants...
                                        #also odd that '3D printing' is split in freq dist and aren't both at full representation...
common_words = []
for terms in mostcommon:
    common_words.append(terms[0])
#Could do some interesting things with the most common words. maybe iteratively go through and see if any company names turn up (simplistic)



#Repeat this process for multiple topics of known interest (example medical, space exploration, construction, etc...)
    #include wikipedia pages for US companies (for now)
    #https://en.wikipedia.org/wiki/List_of_companies_of_the_United_States
    #iterate through each page and get all info and see if any companies are placed within the "3D printing" article.
#Combine all results and cluster articles together. FInd any terms that are of interset.

#from sklearn import _______________
from sklearn.cluster import KMeans #Could do other clustering algo's with other metrics.
from sklearn.pipeline import Pipeline #later
from sklearn.feature_extraction.text import TfidfVectorizer
#from sklearn.selection import RandomizedSearchCV, GridSearchCV



#pipeline.set_params(vect__max_df=0.75,vect_ngram_range=(1,1)).fit(data)             #only looking at unigrams for now, max_df arbitrary


#Futher down the road
vect = TfidfVectorizer(max_df=0.5, max_features=20, min_df=2, stop_words='english') #Number of features is arbitrary for exploration
X = vect.fit_transform(text_list)   #creates sarse matrix

clf = KMeans(n_clusters=10, init='k-means++', max_iter=100, n_init=1, verbose=1)        #arbitrary number
pred = clf.fit_predict(X)

