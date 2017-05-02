#This code applies different algorithms to the tweets processed through tweet_processing.py
#Trying LinearSVC, KNearestNeighbors, Decision Trees, and Random Forests as they are explicitly supported by sci-kit learn's multi-class modules.
            #http://scikit-learn.org/stable/modules/multiclass.html
#Also tried using a fully-connected neural network to predict classes.
#Ultimately found that RandomForestClassifier predicted best, with about a 85% accuracy.
#TBT whether accuracy is truly the best metric; will read up on literature.


import json
import pandas as pd
import re
import nltk
import matplotlib.pyplot as plt
from collections import Counter
from nltk.corpus import stopwords
import string
punctuation = list(string.punctuation)
stop = stopwords.words('english') + punctuation + ['RT', 'via', '…']

from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.model_selection import ShuffleSplit

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import ModelCheckpoint, EarlyStopping
import tweet_processing
            
            
            
path = 'custom_path'
X = tweet_processing.make_data_df(path)
Y = tweet_processing.make_labels(path)

#checking for any missing data.
#total = data.isnull().sum().sort_values(ascending=False)
#percent = (data.isnull().sum()/data.isnull().count()).sort_values(ascending=False)
#missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
#missing_data.head()

#Splitting data into training and cross-validation sets.
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.67, random_state=21)

#Spot-checking various multi-label algorithms
linear_svc = OneVsRestClassifier(LinearSVC(random_state=0)).fit(X_train, Y_train)
pred_linear_svc = linear_svc.predict(X_test)
linear_acc = accuracy_score(Y_test, pred_linear_svc)

KNN = OneVsRestClassifier(KNeighborsClassifier(n_neighbors = 3)).fit(X_train, Y_train)
pred_KNN = KNN.predict(X_test)
KNN_acc = accuracy_score(Y_test, pred_KNN)

Decision_tree = OneVsRestClassifier(DecisionTreeClassifier()).fit(X_train, Y_train)
pred_decision_tree = Decision_tree.predict(X_test)
decision_tree_acc = accuracy_score(Y_test, pred_decision_tree)

random_forest = OneVsRestClassifier(RandomForestClassifier(n_estimators=100)).fit(X_train, Y_train)
pred_random_forest = random_forest.predict(X_test)
random_forest_acc = accuracy_score(Y_test, pred_random_forest)

#Neural Network (Fully Connected)
#Inputs need to be numpy arrays
train = X.values
classes = Y
#Making fully connected neural network. Could implement regularization with dropout, but seems to not be needed
#   as acc and val_acc are similar
nnmodel = Sequential()
nnmodel.add(Dense(208, input_dim=208, kernel_initializer='normal', activation='relu'))
nnmodel.add(Dropout(0.25))
nnmodel.add(Dense(120, kernel_initializer='normal', activation='relu'))
nnmodel.add(Dropout(0.25))
nnmodel.add(Dense(40, kernel_initializer='normal', activation='softmax'))
# Compile model
nnmodel.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

#Checkpointing
filepath='C:/Users/Michael Kang/Desktop/Tweet_nn_analysis_best.hdf5'
#Early = EarlyStopping(monitor='val_acc', min_delta=0, patience=2, verbose=2, mode='auto')
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

#Plot learning curves
nn_model_history = nnmodel.fit(train, classes, validation_split=0.33, epochs=10, batch_size=10, callbacks=callbacks_list)
plt.plot(nn_model_history.history['acc'])
plt.plot(nn_model_history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
#Since just getting an estimate, just getting final value since graph shows it to have mostly stabilized.
nn_score = nn_model_history.history['acc'][-1:][0]

#Now tuning randomforest (which did significantly better in accuracy)

depth = [7,9,15]
maxfeatures = ['auto']#, 'log2', None]
parameters = {'estimator__max_depth': depth, 'estimator__max_features': maxfeatures}
random_forest = OneVsRestClassifier(RandomForestClassifier(n_jobs=-1, random_state=0, n_estimators=100))
random_forest_model = GridSearchCV(random_forest, parameters, cv=10)
random_forest_model.fit(X_train, Y_train)
#print(random_forest_model.best_estimator_)
tuned_random_forest = OneVsRestClassifier(RandomForestClassifier(n_estimators=1000,max_depth=15, max_features='auto', n_jobs=-1, random_state=0)).fit(X_train, Y_train)
pred_tuned_random_forest = tuned_random_forest.predict(X_test)
tuned_random_forest_acc = accuracy_score(Y_test, pred_tuned_random_forest)
#85.297619047619044


'''
#Another python file that contains the function
from plot_learning_curve import plot_learning_curve
title = 'Learning Curves (Random Forest)'
cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)
estimator = OneVsRestClassifier(RandomForestClassifier(n_estimators=1000,max_depth=15, max_features='auto', n_jobs=-1, random_state=0))
plot_learning_curve(estimator, title, X_train, Y_train, cv=cv, n_jobs=-1)
plt.show()
'''
