
#Redacted information as requested.
#It does so by placing the data into a pandas dataframe and processing the data such that data types
#   are correct and anomalous domains are removed.
#Of interest is how since there was not much data - nor many features - natural language processing of
#   the domain names was also conducted, thereby checking if each domain name had tokens that were contained
#   in a created dictionary (of the most common tokens). It is assumed that domain names are related to the
#   type of website it represents, and thus offer insights into the amount spent for that domain.
#
#The various terms will be explained as if one were present at an online auction (one for buying the space/time to 
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
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score, GridSearchCV, train_test_split
from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, ElasticNetCV, LassoCV, LassoLarsCV
from sklearn.linear_model import Lasso, BayesianRidge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import xgboost as xgb
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import ModelCheckpoint, EarlyStopping




#----------------------------------------------------------
"""Processing data and moving it into a pandas dataframe"""
#----------------------------------------------------------
#Reading data into list. Replace file path with custom path.
with open('custom_path.csv', 'r') as f:
    reader = csv.reader(f, dialect='excel', delimiter='\t')
    lst = list(reader)

#Removing tab-delimited characters.
data = []
for line in lst:
    temp = line[0].split('\t')
    data.append(temp)

#Reading data into a pandas dataframe.
headers = data.pop(0)
df = pd.DataFrame(data, columns=headers)





#----------------------------------------------------------
"""Processing dataframe datatypes and dropping irrelevant features"""
#----------------------------------------------------------
#df.dtypes shows that all elements are strings.
#Change numeric features to numeric and round floats to closest cent.
numeric_cols = ['num_seen', 'num_avail', 'num_served', 'num_visible', 'total_spent', 'total_ecp']
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')
df.total_spent = df.total_spent.round(decimals=2)
df.total_ecp = df.total_ecp.round(decimals=2)

#only advertiser for this is _______. Dropping since it doesn't add any additional information
df = df.drop('advertiser', axis=1)
#num_avail has 127/20899 (0.6%) of entries filled out. Dropping feature.
df = df.drop('num_avail', axis=1)





#----------------------------------------------------------
""" Dropping anomalous domains and cleaning domain name strings to tokenize.
    Also taking top 160 most common tokens and making new dataframe 'data'
    Finally adding numerical feautures from df to 'data' """
#----------------------------------------------------------
#Get rid of anomalous domains (domains that dont have served impressions nor a coherent domain name)
df.drop(df.index[[0,1,2,3,12,13,14,15,16,17,27,70,71, -1, -2, -3]], inplace=True)
#Getting rid of quotation marks in domain names
df.loc[df.domain == "'cbssports.com'", 'domain'] = "cbssports.com"
df.loc[df.domain == "'imdb.com'", 'domain'] = "imdb.com"
df.loc[df.domain == "-1.latineuro.com", 'domain'] = "latineuro.com"

#Get rid of signs in front of domain names, and generally cleaning up the names
#Getting rid of .com endings and using word segmentation on the domain names
domains = df.domain.tolist()
words = []
for i in range(0,len(domains)):
    if domains[i].startswith(('.','$')):
        domains[i] = domains[i][1:]
    if domains[i].startswith('&referrer='):
        domains[i] = domains[i][10:]
    if domains[i].endswith('.com'):
        domains[i] = domains[i][:-4]
    words.append(segment(domains[i]))
df['domain'] = pd.Series(domains)

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
data['num_seen'] = df['num_seen']
data['served'] = df['num_served']
data['num_visible'] = df['num_visible']
data['total_ecp'] = df['total_ecp']
data['total_spent'] = df['total_spent']
#Dropping any domains that don't have any served impressions.
#In hindsight, for larger datasets, it would be wiser to drop these before tokenizing, but for
#   a preliminary analysis, it should be sufficient.
data = data[data.total_ecp != 0]
data = data[pd.notnull(data['total_ecp'])]





#----------------------------------------------------------
""" Applying regression techniques to create a predictive model..."""
#----------------------------------------------------------
X = #Redacted information as requested.
Y = #Redacted information as requested.

#We split our data to have unseen data to test against later.
#   Could have just done cross-validation on the whole set, but this is better practice.
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, train_size=.8)

#Since not all of our token features should be important, trying out lasso and elasticNet first
#Also trying ridge, SVR, ensembles, and boosting.
#Spot-checking not shown.

#Lasso
alphas=[0.3, 0.1, 0.03, 0.001, 0.003, 0.0001,1,2,3,4,5,6,7,8,9,10,11,12]
lasso = Lasso(random_state=17)
lasso_model = GridSearchCV(estimator=lasso, param_grid=dict(alpha=alphas), cv=10)
lasso_model.fit(X_train, Y_train)
#print(lasso_model.best_score_)
#print(lasso_model.best_estimator_.alpha)
#print(lasso_model.best_estimator_)
lasso = Lasso(alpha = 1, random_state=17).fit(X_train, Y_train)
score_lasso = lasso.score(X_test, Y_test)


#Elastic Net
ENSTest = ElasticNetCV(alphas=[0.0001,0.0003, 0.0005, 0.001, 0.03, 0.01, 0.3, 0.1, 3, 1, 10, 30], l1_ratio=[.01, 0.3, .1, .3, .5, .9, .99], max_iter=5000, random_state=3).fit(X_train, Y_train)
score_EN = ENSTest.score(X_test, Y_test)


#Ridge
alphas=[0.1,0.001,0.0001,1,2,3,4,5,6,7,8,9,10,11,12,15]
ridge = Ridge(random_state=2)
ridge_model = GridSearchCV(estimator=ridge, param_grid=dict(alpha=alphas))
ridge_model.fit(X_train, Y_train)
#print(ridge_model.best_score_)
#print(ridge_model.best_estimator_.alpha)
#print(ridge_model.best_estimator_)
ridge = Ridge(alpha = 15, random_state=2).fit(X_train, Y_train)
score_ridge = ridge.score(X_test, Y_test)


#Random Forest
random_forest = RandomForestRegressor(n_estimators=2900, random_state=11).fit(X_train, Y_train)
score_forest = random_forest.score(X_test, Y_test)

#Gradient Boosting
GBest = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05, max_depth=3, max_features='sqrt',
                                               min_samples_leaf=15, min_samples_split=10, loss='huber', random_state=5).fit(X_train, Y_train)
score_GBest = GBest.score(X_test, Y_test)

#XGBoosting
parameters= {'max_depth': [2,4,6,7,8], 'n_estimators': [50,100,200], 'learning_rate': [0.05,0.1,0.3]}
xgb = xgb.XGBRegressor(seed=7)
xgb_model = GridSearchCV(xgb, parameters, n_jobs=1, cv=10)
xgb_model.fit(X_train, Y_train)
#print(ridge_model.best_estimator_). Note: increased n_estimators.
model_xgb = xgb.XGBRegressor(n_estimators=360, max_depth=2, learning_rate=0.05, seed=7) #the params were tuned using xgb.cv
model_xgb.fit(X_train, Y_train)
score_xgb = model_xgb.score(X_test, Y_test)

#Neural Network (Fully Connected)
#Inputs need to be numpy arrays
temp_data = data.values
train = temp_data[:,1:].astype(float)
features = temp_data[:,0]

#Making fully connected neural network. Could implement regularization with dropout, but seems to not be needed
#   as acc and val_acc are similar
nnmodel = Sequential()
nnmodel.add(Dense(80, input_dim=164, kernel_initializer='normal', activation='relu'))
nnmodel.add(Dense(20, kernel_initializer='normal', activation='relu'))
nnmodel.add(Dense(1, kernel_initializer='normal'))
# Compile model
nnmodel.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

#Checkpointing
filepath='C:/Users/Michael Kang/Desktop/RockerboxRegressionBest.hdf5'
#Early = EarlyStopping(monitor='val_acc', min_delta=0, patience=2, verbose=2, mode='auto')
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

#Plot learning curves
nn_model_history = nnmodel.fit(train, features, validation_split=0.25, epochs=100, batch_size=10, callbacks=callbacks_list)
plt.plot(nn_model_history.history['acc'])
plt.plot(nn_model_history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
#Since just getting an estimate, just getting final value since graph shows it to have mostly stabilized.
nn_score = nn_model_history.history['acc'][-1:][0]

#Just making a nice visual to compare accuracy.
models = pd.DataFrame({
    'Model': ['Lasso', 'Elastic Net', 'Ridge', 'Random Forest', 'Gradient Boosting', 'XGB', 'Neural Network'],
    'Score': [score_lasso, score_EN, score_ridge, score_forest, score_GBest, score_xgb, nn_score]})
print(models.sort_values(by='Score', ascending=False))





#combining the results of XGBoost, the neural network, and Random Forests, which had the best results.
#Run this last with X_provided being a processed and tokenized domain. Outputs predicted total_spent.
def predict(X_provided):
    random_forest = RandomForestRegressor(n_estimators=2900, random_state=11).fit(X, Y)
    pred_random_forest = random_forest.predict(X_provided)
    
    Y_index = nnmodel.predict(X_provided)
    pred_neural = np.argmax(Y_index,axis=1)
    
    xgbost = xgb.XGBRegressor(n_estimators=360, max_depth=2, learning_rate=0.05, seed=7).fit(X, Y)
    pred_xgb = xgbost.predict(X_provided)
    return (pred_random_forest + pred_xgb + pred_neural)/3

