# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 11:59:33 2017

@author: Michael Kang
"""

#This program seeks to classify which domains' impressions will be served given domain name and num_seen.
#It does so by placing the data into a pandas dataframe and processing the data sch that data types
#   are correct and anomalous domains are removed.
#Of interest is how since there was not much data - nor many features - natural language processing of
#   the domain names was also conducted, thereby checking if each domain name had tokens that were contained
#   in a created dictionary (of the most common tokens). It is assumed that domain names are related to the
#   type of website it represents, and thus offer insights into the amount spent for that domain.

#the various terms will be explained as if one were present at an online auction (one for buying the space/time to 
#   show your online ad)
#
#advertiser:        identity of advertiser
#domain:            website url (any given auction)
#num_seen:          number of impressions seen (seeing an item at the auction)
#num_avail:         number of impressions typically available at that domain. Not usually filled out.
#num_served:        number of impressions served (number of items bought/won at the auction)
#num_visible:       number of impressions actually visible on website (a percentage of num_served)
#total_spent:       the amount of money spent at an auction
#total_ecp:         "An estimate of a bid that is likely to win the impression from a given publisher based on 
#                           observing historical bids." Calculated using AppNexus.


#Importing all necesssary packages
import csv
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from wordsegment import segment
import nltk

from sklearn.grid_search import GridSearchCV
from sklearn.svm import LinearSVC, SVC
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score, train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import AdaBoostClassifier
import xgboost as xgb





#----------------------------------------------------------
"""Processing data and moving it into a pandas dataframe"""
#----------------------------------------------------------
#Now just reading into list
with open('custom_input_path.csv', 'r') as f:
    reader = csv.reader(f, dialect='excel', delimiter='\t')
    lst = list(reader)

#Could also have used regexpressions, I think?
data = []
for line in lst:
    temp = line[0].split('\t')
    data.append(temp)

#Reading data into a pandas dataframe
headers = data.pop(0)
df = pd.DataFrame(data, columns=headers)





#----------------------------------------------------------
""" Processing dataframe datatypes and dropping irrelevant features
    Feature Engineering: feature of "if served or not"""
#----------------------------------------------------------
#df.dtypes shows that all elements are strings.
#Change numeric features to numeric and round floats to closest cent
numeric_cols = ['num_seen', 'num_avail', 'num_served', 'num_visible', 'total_spent', 'total_ecp']
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')
df.total_spent = df.total_spent.round(decimals=2)
df.total_ecp = df.total_ecp.round(decimals=2)

#Checking for correlations between numerical features to get a better feel for the data
corrmat = df.corr()
plt.subplots(figsize=(8,6))
sns.heatmap(corrmat, vmax=0.9, square=True)

#only advertiser for this is baublebar. Dropping since it doesn't add any information
df = df.drop('advertiser', axis=1)
#num_avail has 127/20899 (0.6%) of entries filled out. Dropping feature.
df = df.drop('num_avail', axis=1)

#Trying to predict which impressions will be served.
served_df = df[['domain', 'num_seen', 'num_served']]

#Making a new feature of whether of not the seen impression ended up served
served_df['Served'] = served_df['num_served'].fillna(0)
served_df.loc[served_df.Served != 0, 'Served'] = 1
served_df = served_df.drop('num_served', axis=1)





#----------------------------------------------------------
""" Dropping anomalous domains and cleaning domain name strings to tokenize.
    Also taking top 160 most common tokens and making new dataframe 'data'
    Finally adding numerical feautures from df to 'data' """
#----------------------------------------------------------
#Get rid of anomalous domains
served_df.drop(served_df.index[[0,1,2,3,12,13,14,15,16,17,27,70,71, -1, -2, -3]], inplace=True)
#Getting rid of quotation marks
served_df.loc[served_df.domain == "'cbssports.com'", 'domain'] = "cbssports.com"
served_df.loc[served_df.domain == "'imdb.com'", 'domain'] = "imdb.com"
served_df.loc[served_df.domain == "-1.latineuro.com", 'domain'] = "latineuro.com"


#Get rid of signs in front of domain names, and generally cleaning up the names
#Getting rid of .com endings and using word segmentation on the domain names
domains = served_df.domain.tolist()
words = []
for i in range(0,len(domains)):
    if domains[i].startswith(('.','$')):
        domains[i] = domains[i][1:]
    if domains[i].startswith('&referrer='):
        domains[i] = domains[i][10:]
    if domains[i].endswith('.com'):
        domains[i] = domains[i][:-4]
    words.append(segment(domains[i]))
served_df['domain'] = pd.Series(domains)


#Take top 160 features, with the cutoff being that the token is present in about at least 40 domain-names
wordlist = []
for blurbs in words:
    wordlist.extend(blurbs)

wordlist = nltk.FreqDist(wordlist)
mostcommon = wordlist.most_common(160)
common_words = []
for terms in mostcommon:
    common_words.append(terms[0])

features = []
for i in range(0,len(words)):               #for each set of words from each domainname
    word = set(words[i])
    temp = {}
    for feat in common_words:              #for each token from the word_features
        temp[feat] = (feat in word)
    features.append(dict(temp))

data = pd.DataFrame(features)
data = data.dropna()

#Now we append columns from served_df to data
data['num_seen'] = served_df['num_seen']
data['served'] = served_df['Served']
#Note that an impression is served only 15% of the time. Skewed classes





#----------------------------------------------------------
""" Applying classification techniques to create a predictive model for whether served or not
    Here I spotcheck"""
#----------------------------------------------------------
#Could use train_test_split, but this split isn't needed and we need all the data we can get
X_train = data.drop('served', axis=1)
Y_train = data['served']

linear_svc = LinearSVC()
cv = StratifiedKFold(n_splits=5,shuffle=True,random_state=45)
linear_score = cross_val_score(linear_svc, X_train, Y_train, cv=cv, scoring='f1').mean()

svc = SVC(kernel='rbf')
cv = StratifiedKFold(n_splits=5,shuffle=True,random_state=45)
svc_score = cross_val_score(svc, X_train, Y_train, cv=cv, scoring='f1').mean()

knn = KNeighborsClassifier(n_neighbors = 3)
cv = StratifiedKFold(n_splits=5,shuffle=True,random_state=45)
knn_score = cross_val_score(knn, X_train, Y_train, cv=cv, scoring='f1').mean()

gaussian = GaussianNB()
cv = StratifiedKFold(n_splits=5,shuffle=True,random_state=45)
gaussian_score = cross_val_score(gaussian, X_train, Y_train, cv=cv, scoring='f1').mean()

perceptron = Perceptron()
cv = StratifiedKFold(n_splits=5,shuffle=True,random_state=45)
perceptron_score = cross_val_score(perceptron, X_train, Y_train, cv=cv, scoring='f1').mean()

sgd = SGDClassifier()
cv = StratifiedKFold(n_splits=5,shuffle=True,random_state=45)
sgd_score = cross_val_score(sgd, X_train, Y_train, cv=cv, scoring='f1').mean()

decision_tree = DecisionTreeClassifier()
cv = StratifiedKFold(n_splits=5,shuffle=True,random_state=45)
Dtree_score = cross_val_score(decision_tree, X_train, Y_train, cv=cv, scoring='f1').mean()

random_forest = RandomForestClassifier(n_estimators=100)
cv = StratifiedKFold(n_splits=5,shuffle=True,random_state=45)
random_forest_score = cross_val_score(random_forest, X_train, Y_train, cv=cv, scoring='f1').mean()

adaboost = AdaBoostClassifier(n_estimators=100)
cv = StratifiedKFold(n_splits=5,shuffle=True,random_state=45)
ada_score = cross_val_score(adaboost, X_train, Y_train, cv=cv, scoring='f1').mean()

XGB = xgb.XGBClassifier()
cv = StratifiedKFold(n_splits=5,shuffle=True,random_state=45)
xgb_score = cross_val_score(XGB, X_train, Y_train, cv=cv, scoring='f1').mean()



#Also looking at precision in addition to F1 score because data is skewed (85no:15yes ratio)
a = cross_val_score(linear_svc, X_train, Y_train, cv=cv, scoring='precision').mean()
b = cross_val_score(svc, X_train, Y_train, cv=cv, scoring='precision').mean()
c = cross_val_score(knn, X_train, Y_train, cv=cv, scoring='precision').mean()
d = cross_val_score(gaussian, X_train, Y_train, cv=cv, scoring='precision').mean()
e = cross_val_score(perceptron, X_train, Y_train, cv=cv, scoring='precision').mean()
f = cross_val_score(sgd, X_train, Y_train, cv=cv, scoring='precision').mean()
g = cross_val_score(decision_tree, X_train, Y_train, cv=cv, scoring='precision').mean()
h = cross_val_score(random_forest, X_train, Y_train, cv=cv, scoring='precision').mean()
i = cross_val_score(adaboost, X_train, Y_train, cv=cv, scoring='precision').mean()
j = cross_val_score(XGB, X_train, Y_train, cv=cv, scoring='precision').mean()


#Just making a nice visual to compare metrics.
models = pd.DataFrame({
    'Model': ['linear SVC', 'SVC', 'KNN', 'gaussian', 'perceptron', 'SGD', 'decision tree', 'random forest', 'Adaboost', 'XGB'],
    'F1': [linear_score, svc_score, knn_score, gaussian_score, perceptron_score, sgd_score, Dtree_score, random_forest_score, ada_score, xgb_score],
    'Precision': [a,b,c,d,e,f,g,h,i,j]})
print(models.sort_values(by='Precision', ascending=False))





#----------------------------------------------------------
""" Choosing and tuning prospective algorithms"""
#----------------------------------------------------------
#Going to work with KNN, XGB, and SVC, as they boast >0.4 F1 and >0.7 precision
#KNearestNeighbors
k = np.arange(20)+1
parameters = {'n_neighbors': k}
knn = KNeighborsClassifier()
knn_model = GridSearchCV(knn,parameters,scoring = 'f1',cv=10)
knn_model.fit(X_train, Y_train)
#print(knn_model.best_estimator_)
knn = KNeighborsClassifier(n_neighbors = 5)
cv = StratifiedKFold(n_splits=5,shuffle=True,random_state=45)
knn_score = cross_val_score(knn, X_train, Y_train, cv=cv, scoring='f1').mean()

#SVC
Cs = [0.001, 0.01, 0.1, 1, 10]
gammas = [0.001, 0.01, 0.1, 1]
parameters = {'C': Cs, 'gamma' : gammas}
svc = SVC(kernel='rbf')
svc_model = GridSearchCV(svc,parameters,scoring = 'f1',cv=10)
svc_model.fit(X_train, Y_train)
#print(svc_model.best_estimator_)
svc = SVC(kernel='rbf', C=1, gamma=0.001)
cv = StratifiedKFold(n_splits=5,shuffle=True,random_state=45)
svc_score = cross_val_score(svc, X_train, Y_train, cv=cv, scoring='f1').mean()


#XGBoosting
xgb_model = GridSearchCV(cv=5, error_score='raise',
       estimator=xgb.XGBClassifier(base_score=0.5, colsample_bylevel=1, colsample_bytree=0.8,
       gamma=0, max_delta_step=0,
       missing=None, n_estimators=1000, nthread=-1,
       objective='binary:logistic', reg_alpha=0, reg_lambda=1,
       scale_pos_weight=1, seed=0, silent=True),
       fit_params={}, iid=True, n_jobs=-1, 
       param_grid={'min_child_weight': [1, 3, 5], 'max_depth': [3, 5, 7],
                   'learning_rate': [0.01,0.05,0.1], 'subsample':[0.7,0.8,1.0,1.2]
                   },
       pre_dispatch='2*n_jobs', refit=True, scoring='f1', verbose=0)
xgb_model.fit(X_train, Y_train)
#print(xgb_model.grid_scores_)              min_child_weight:5. max_depth=3
best_params = {'learning_rate': 0.05, 'subsample': 0.8, 'min_child_weight': 5, 'max_depth': 3}
dtrain = xgb.DMatrix(X_train, label = Y_train)
#Can't use F1 metric because its then comparing nonbinary with binary targets
#Instead using log-loss since its not as subject to skewed classes
#early stopping to avoid overfitting
best_model = xgb.cv(best_params, dtrain,  num_boost_round=500, early_stopping_rounds=20, metrics= 'logloss', nfold=5)
best_model.loc[:,["test-logloss-mean", "train-logloss-mean"]].plot()
#Stopped after 106 iterations.
#If we change the metric to error and early_stopping_rounds=20, then 47 iterations and accuracy=88%.
#Looking purely at precision, this is an improvement from when untuned! (77.87% -->88%)
#But still using logloss since precision isn't a good metric.

best_xgb = xgb.XGBClassifier(learning_rate=0.05, max_depth=3, min_child_weight=5, subsample=0.8, seed=7) #the params were tuned using xgb.cv
cv = StratifiedKFold(n_splits=5,shuffle=True,random_state=45)
best_xgb_score = cross_val_score(svc, X_train, Y_train, cv=cv, scoring='f1').mean()

#K-Nearest Neighbors shows the highest F1 score.
tuned_models = pd.DataFrame({
    'Model': ['K-Nearest Neighbors', 'SVC', 'XGBoosting'],
    'F1': [knn_score, svc_score, best_xgb_score]})
print(tuned_models.sort_values(by='F1', ascending=False))




#Since K-Nearest Neighbors shows the highest F1 score, using those results
#Run this last with X_provided being a processed and tokenized domain. Outputs whether served(1) or not(1).
def predict(X_provided):
    knn = KNeighborsClassifier(n_neighbors = 5).fit(X_train, Y_train)
    pred_KNN = knn.redict(X_provided)
    return pred_KNN
