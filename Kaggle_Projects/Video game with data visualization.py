import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats

from sklearn.model_selection import KFold
from sklearn import linear_model
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn import svm
from sklearn.ensemble import AdaBoostRegressor
from sklearn.model_selection import cross_val_score

from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint

import warnings
warnings.filterwarnings('ignore')

train = pd.read_csv('C:/Users/Michael Kang/Desktop/Data_Files/vgsales.csv')

#train['Global_Sales'].describe()
#sns.distplot(train['Global_Sales'])
#print("Skewness: %f" % train['Global_Sales'].skew())         Skewness: 17.400645
#print("Kurtosis: %f" % train['Global_Sales'].kurt())         Kurtosis: 603.932346
"""
#Correlation between Global_Sales and Year  
data = pd.concat([train['Global_Sales'], train['Year']], axis=1)
data.plot.scatter(x='Year', y='Global_Sales', ylim=(0,100));
                 
#Relationship between Global_Sales and Genre
data = pd.concat([train['Global_Sales'], train['Genre']], axis=1)
fig = sns.boxplot(x='Genre', y='Global_Sales', data=data)

#Relationship between Global_Sales and Platform
data = pd.concat([train['Global_Sales'], train['Platform']], axis=1)
fig = sns.boxplot(x='Platform', y='Global_Sales', data=data)

#Relationship between Global_Sales and Publisher
data = pd.concat([train['Global_Sales'], train['Publisher']], axis=1)
fig = sns.boxplot(x='Publisher', y='Global_Sales', data=data)


#correlation matrix
corrmat = train.corr()
sns.heatmap(corrmat, vmax=.8, square=True);
           
#Scatterplots not needed as only one of our features is numeric
"""

#*****************************************************************

#Dropping Features
train = train.drop(['Rank', 'Name'], axis=1)

#Fixing Missing Data
total = train.isnull().sum().sort_values(ascending=False)
percent = (train.isnull().sum()/train.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(9)


train = train.drop((missing_data[missing_data['Total'] > 60]).index,1)    #This drops the feature with missing data ('Year' in this case)
'''train = train.drop(train.loc[train['Publisher'].isnull()].index)           #This drops all instaces of NaN in feature 'Publisher' '''
train.isnull().sum().max()       #to check if any missing data remains



#*******************************************************************
#Now looking for Outliers!

#Standardizing data
Global_Sales_Scaled = StandardScaler().fit_transform(train['Global_Sales'][:,np.newaxis]);
low_range = Global_Sales_Scaled[Global_Sales_Scaled[:,0].argsort()][:10]
high_range= Global_Sales_Scaled[Global_Sales_Scaled[:,0].argsort()][-15:]
print('outer range (low) of the distribution:')
print(low_range)
print('\nouter range (high) of the distribution:')
print(high_range)

#Deleting the outlier
train = train.drop(train[train['Global_Sales'] == 82.74].index)   #This is WiiSports, which comes with sales of the Nintendo Wii


#*************************************************************************
#Looking For Normality

#Looking at Global_Sales
sns.distplot(train['Global_Sales'], fit=norm)
res = stats.probplot(train['Global_Sales'], plot=plt)
#Log transform
train['Global_Sales'] = np.log(train['Global_Sales'])

sns.distplot(train['Global_Sales'], fit=norm)
res = stats.probplot(train['Global_Sales'], plot=plt)


train['Has_NA_Sales'] = pd.Series(len(train['NA_Sales']), index=train.index)
train['Has_NA_Sales'] = 0 
train.loc[train['NA_Sales']>0,'Has_NA_Sales'] = 1
#Transform Data
train.loc[train['Has_NA_Sales']==1,'NA_Sales'] = np.log(train['NA_Sales'])

#Will figure out a better way of going about this
sns.distplot(train['NA_Sales'], fit=norm)
res = stats.probplot(train['NA_Sales'], plot=plt)

#Dropping Has_NA_Sales
train = train.drop(['Has_NA_Sales'], axis=1)

#********************************************************************

#Getting ready to plug into models
X_train = train.drop('Global_Sales', axis=1)
Y_train = train['Global_Sales']

#convert categorical variables into dummies
X_train = pd.get_dummies(X_train)

#***********************************************************************

ridge = linear_model.Ridge()
cv = KFold(n_splits=5,shuffle=True,random_state=42)
results_ridge = cross_val_score(ridge, X_train, Y_train, cv=cv)
print('Ridge Results: %.2f%% +/-%.2f%%' % (results_ridge.mean()*100, results_ridge.std()*100))
results_ridge = results_ridge.mean()*100

B_ridge = linear_model.BayesianRidge()
cv = KFold(n_splits=5,shuffle=True,random_state=42)
results_B_ridge = cross_val_score(B_ridge, X_train, Y_train, cv=cv)
print('B_ridge Results: %.2f%% +/-%.2f%%' % (results_B_ridge.mean()*100, results_B_ridge.std()*100))
results_B_ridge = results_B_ridge.mean()*100

huber = linear_model.HuberRegressor()
cv = KFold(n_splits=5,shuffle=True,random_state=42)
results_huber = cross_val_score(huber, X_train, Y_train, cv=cv)
print('Huber Results: %.2f%% +/-%.2f%%' % (results_huber.mean()*100, results_huber.std()*100))
results_huber = results_huber.mean()*100

lasso = linear_model.Lasso(alpha=1e-4)
cv = KFold(n_splits=5,shuffle=True,random_state=42)
results_lasso = cross_val_score(lasso, X_train, Y_train, cv=cv)
print('Lasso Results: %.2f%% +/-%.2f%%' % (results_lasso.mean()*100, results_lasso.std()*100))
results_lasso = results_lasso.mean()*100

bag = BaggingRegressor()
cv = KFold(n_splits=5,shuffle=True,random_state=42)
results_bag = cross_val_score(bag, X_train, Y_train, cv=cv)
print('Bagging Results: %.2f%% +/-%.2f%%' % (results_bag.mean()*100, results_bag.std()*100))
results_bag = results_bag.mean()*100

forest = RandomForestRegressor()
cv = KFold(n_splits=5,shuffle=True,random_state=42)
results_forest = cross_val_score(forest, X_train, Y_train, cv=cv)
print('Random Forest Results: %.2f%% +/-%.2f%%' % (results_forest.mean()*100, results_forest.std()*100))
results_forest = results_forest.mean()*100

ada = AdaBoostRegressor()
cv = KFold(n_splits=5,shuffle=True,random_state=42)
results_ada = cross_val_score(ada, X_train, Y_train, cv=cv)
print('Ada Results: %.2f%% +/-%.2f%%' % (results_ada.mean()*100, results_ada.std()*100))
results_ada = results_ada.mean()*100

X_train = X_train.values
Y_train = Y_train.values

#Creating model
model = Sequential()
model.add(Dense(256, input_dim=625, kernel_initializer='normal', activation='relu'))
model.add(Dense(32, kernel_initializer='normal', activation='relu'))
model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
# Compile model
model.compile(loss='mse', optimizer='adam', metrics = ['accuracy'])
#Checkpointing
filepath='C:/Users/Michael Kang/Desktop/Data_Files/Housing/weights.best.hdf5'
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]
#Plot learning curves
neural_history = model.fit(X_train, Y_train, validation_split=0.33, nb_epoch=15, batch_size=5, callbacks=callbacks_list)
plt.plot(neural_history.history['acc'])
plt.plot(neural_history.history['val_acc'])
plt.title('neural model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')

neural_score = model.evaluate(X_train, Y_train)[1]


models = pd.DataFrame({
    'Model': ['Ridge', 'Bayesian Ridge', 'Huber', 'Lasso', 'Bagging', 'Random Forest', 'AdaBoost', 'Neural Network'],
    'Score': [results_ridge, results_B_ridge, results_huber, results_lasso, results_bag, results_forest, results_ada, neural_score]})
print(models.sort_values(by='Score', ascending=False))
