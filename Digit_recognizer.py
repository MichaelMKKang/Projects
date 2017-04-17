# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 21:20:33 2017

@author: Michael Kang
"""
#This is practice code which seeks to recognize digits from the MNIST dataset
#primarily through convolutional neural networks.
#This was used to create a submission to Kaggle.com, a platform
#for data science competitions. This particular submission (having trained over
#additional epochs) got at the 8th percentile.

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage.filters import gaussian_filter

from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

from sklearn.linear_model import SGDClassifier

from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.callbacks import ModelCheckpoint, EarlyStopping




#Data is of 28x28 greyscale image. train has 1 more feature 'label' to indicate #. Other features are labeled pixel0, pixel1, pixelX
train = pd.read_csv('C:/Users/Michael Kang/Desktop/Data_Files/Digit Recognizer/Digit_train.csv')
test= pd.read_csv('C:/Users/Michael Kang/Desktop/Data_Files/Digit Recognizer/Digit_test.csv')

X_train_df = train.drop('label', axis=1)
Y_train_df = train['label']
X_test_df = test
X_train_df.shape, Y_train_df.shape, X_test_df.shape

#******************************************************************************************************
#Getting a look at the data
i=50
img=X_train_df.iloc[i].as_matrix()
img=img.reshape((28,28))
img=gaussian_filter(img, sigma=1)               #I'm not sure if this helps that much...
plt.imshow(img,cmap='gray')
plt.title(X_train_df.iloc[i,0])

#*****************************************************************************************************
#This is SGD with just unprocessed data. Note that SVM takes way too long to run
sgd = SGDClassifier()                                           #SGD Results: 86.41% +/-1.08%
cv = KFold(n_splits=5,shuffle=True,random_state=42)
results = cross_val_score(sgd, X_train_df, Y_train_df, cv=cv)
print("SGD Results: %.2f%% +/-%.2f%%" % (results.mean()*100, results.std()*100))

#Manually splitting training set into cross-validation sets using train_test_split
X_train_df_val, X_test_df_val, Y_train_df_val, Y_test_df_val = train_test_split(X_train_df, Y_train_df, train_size = 0.75, random_state = 46)
X_train_df_val.shape, X_test_df_val.shape, Y_train_df_val.shape, Y_test_df_val.shape

#checking to see if using accuracy_score and train_test_split will have a significant effect.
sgd = SGDClassifier()                                           #SGD Results: 89.20%
sgd.fit(X_train_df_val, Y_train_df_val)                         #No really significant difference.
Y_pred = sgd.predict(X_test_df_val)
acc_1 = round(accuracy_score(Y_test_df_val, Y_pred)*100, 2)

#********************************************************************************************************
#SVC and Kneighbors both take too long to fit. Will try simplifying the data by binarizing.
X_train_df_binary = X_train_df
X_test_df_binary = X_test_df
X_train_df_binary[X_train_df_binary>0]=1
X_test_df_binary[X_test_df_binary>0]=1
#Looking at the data again
img=X_train_df_binary.iloc[i].as_matrix()
img=img.reshape((28,28))
img=gaussian_filter(img, sigma=1)
plt.imshow(img,cmap='gray')
plt.title(X_train_df_binary.iloc[i,0])
#histogram of values for index50 show that all pixels are either black or white
plt.hist(X_train_df_binary.iloc[i])

#Now fitting the binarized data into SGD, SVM, and Kneighbors
sgd = SGDClassifier()                                           #SGD Results: 88.17% +/-0.42%
cv = KFold(n_splits=5,shuffle=True,random_state=42)
results = cross_val_score(sgd, X_train_df_binary, Y_train_df, cv=cv)
print('SGD Results: %.2f%% +/-%.2f%%' % (results.mean()*100, results.std()*100))

#Still, fitting SVM and KNeighbors using cross_val_score takes too long. Trying using .score/train_test_split
#Again manually splitting training set into cross-validation sets using train_test_split
X_train_df_val_bin, X_test_df_val_bin, Y_train_df_val_bin, Y_test_df_val_bin = train_test_split(X_train_df_binary, Y_train_df, train_size = 0.75, random_state = 46)
X_train_df_val_bin.shape, X_test_df_val_bin.shape, Y_train_df_val_bin.shape, Y_test_df_val_bin.shape


#*************************************************************************
#Now working with neural networks. To have reproducibility, we set randomizer seed
seed = 7
np.random.seed(seed)

#Keras requires inputs to be numpy arrays, not pandas dataframes.
temp_data = train.values

X_train = temp_data[:,1:].astype(float)
Y_train = temp_data[:,0]
X_test = test.values

#Another look at the data
X_image = X_train.reshape(42000, 28,28)
for i in range(0, 9):
	plt.subplot(330 + 1 + i)
	plt.imshow(X_image[i], cmap=plt.get_cmap('gray'))
# show the plot
plt.show()

#Noapplying one hot encoding to numpy array Y_train
#using keras packaging since it seems to work better without indexing errors...
encoder = LabelEncoder()
encoder.fit(Y_train)
dummy_y = np_utils.to_categorical(Y_train)
#Y_train.shape is now (42000, 10)



model = Sequential()              #Neural Network Results: 96.66% +/-0.26%
model.add(Dense(200, input_dim=784, kernel_initializer='normal', activation='relu'))
model.add(Dense(30, kernel_initializer='normal', activation='relu'))
model.add(Dense(10, kernel_initializer='normal', activation='softmax'))
# Compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    

#Checkpointing
filepath='C:/Users/Michael Kang/Desktop/Data_Files/Housing/weights.best.hdf5'
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

#Plot learning curves
model_history = model.fit(X_train, dummy_y, validation_split=0.33, epochs=15, batch_size=10, callbacks=callbacks_list)
plt.plot(model_history.history['acc'])
plt.plot(model_history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')


'''     #This is another way to evaluate the model
create_baseline()                                               #This yielded 97.64%
model.fit(X_train, dummy_y, epochs = 10, batch_size=10)
scores = model.evaluate(X, Y)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
'''

#If we were to use this neural network, would submit using the following code
'''create_baseline()
model.fit(X, Y, nb_epoch=10, batch_size=5,  verbose=2)
Y_pred = model.predict(X_test)
Y_index = np.argmax(Y_pred,axis=1)

submission = pd.DataFrame({
        'ImageId': (np.arange(Y_index.shape[0])+1),
        'Label': Y_index
    })
submission.to_csv('../output/submission.csv', index=False)
'''

#***************************************************************************************
#Now using a Convoluted Neural Netwok with feeds being the images themselves
X_train_2D = X_train.reshape(X_train.shape[0], 28, 28, 1)      #Thus the images have dimensions 1x28x28 (depth 1 since no color)
#plt.imshow(X_train_2D[3],cmap='gray').     #The image was reshaped correctly
X_test_2D = X_test.reshape(X_test.shape[0], 28, 28, 1)
#Now normalize each value between 0 and 1 by dividing by 255.
    #Will have to learn more about #standardScaler and pipeline, since I didn't get high accuracy when used on the above neural network
X_train_2D = X_train_2D / 255
X_test_2D = X_test_2D / 255

X_trained_2D, X_val_2D, Y_trained_2D, Y_val_2D = train_test_split(X_train_2D, dummy_y, train_size=0.7, random_state=seed)

#Creating & Compiling Model
#*************************LOOK AT HOW I IMPLEMENTED ON KAGGLE MNIST NOTEBOOK
model = Sequential()                                                            #These dimensions are obtained using print(model.output_shape)
model.add(Convolution2D(32, (3, 3), activation='relu', input_shape=(28,28,1)))    #(None, 26, 26, 32)
model.add(Convolution2D(32, (3, 3), activation='relu'))                           #(None, 24, 24, 32)
model.add(MaxPooling2D(pool_size=(2,2)))                                        #(None, 12, 12, 32)
model.add(Dropout(0.25))
model.add(Flatten())                                                            #(None, 4608)

model.add(Dense(128, activation='relu'))                                        #(None, 288)
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))                                         #(None, 10)
#Adding augmentation
data_generated = ImageDataGenerator(zoom_range = 0.1, height_shift_range = 0.1, width_shift_range = 0.1,  rotation_range = 15)
#Compiling now
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

#Checkpointing
filepath='C:/Users/Michael Kang/Desktop/Data_Files/Housing/convoluted_weights.best.hdf5'
Early = EarlyStopping(monitor='val_acc', min_delta=0, patience=2, verbose=2, mode='auto')
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [Early, checkpoint]
#callbacks_list = [checkpoint]

#Plot learning curves
conv_model_history = model.fit_generator(data_generated.flow(X_trained_2D, Y_trained_2D ,batch_size=32), steps_per_epoch = X_trained_2D.shape[0], epochs = 50, verbose=1, validation_data=(X_val_2D, Y_val_2D), callbacks=callbacks_list)
plt.plot(conv_model_history.history['acc'])
plt.plot(conv_model_history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')

'''
Y_pred = model.predict(X_test_2D,  batch_size=32)
y_index = np.argmax(Y_pred,axis=1)
submission = pd.DataFrame({
        'ImageId': (np.arange(y_index.shape[0])+1),
        'Label': y_index
    })
submission.to_csv('C:/Users/Michael Kang/Desktop/Data_Files/Digit Recognizer/3rd try.csv', index=False)
'''            






