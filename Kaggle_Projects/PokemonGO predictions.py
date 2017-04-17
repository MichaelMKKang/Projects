import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
from datetime import datetime
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import log_loss
from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.kernel_approximation import RBFSampler


from keras.models import Sequential
from keras.layers import Dense

from sklearn.model_selection import GridSearchCV
#from mpl_toolkits.basemap import Basemap
#from matplotlib import animation

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn import linear_model
from sklearn.metrics import make_scorer
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn import svm
from sklearn.metrics import r2_score
from sklearn.ensemble import AdaBoostRegressor
from sklearn.model_selection import cross_val_score

import warnings
warnings.filterwarnings('ignore')

train = pd.read_csv('C:/Users/Michael Kang/Desktop/Data_Files/300k.csv', low_memory=False)

#sns.distplot(train['pokemonId'])

#No missing data
NAs = pd.concat([train.isnull().sum()], axis=1)
NAs[NAs.sum(axis=1) > 0]

#I dont really know how S2 cells work right now. I only kind of understand the idea
train = train.drop(['_id', 'cellId_90m', 'cellId_180m', 'cellId_370m', 'cellId_730m', 'cellId_1460m', 'cellId_2920m', 'cellId_5850m'],1)
#Dropping booleans regarding if there is a gym/pokestop within the closest X meters in favor of using gym/pokestop distance in km.
train = train.drop(['gymIn100m', 'gymIn250m', 'gymIn500m', 'gymIn1000m', 'gymIn2500m', 'gymIn5000m', 'pokestopIn100m', 'pokestopIn250m', 'pokestopIn500m', 'pokestopIn1000m', 'pokestopIn2500m', 'pokestopIn5000m'],1)
#Dropping and appearedDayofWeek since that information is contained in other features. Checked and pokemon go does not have bonuses based on which day of the week it is.
train = train.drop(['appearedDayOfWeek'],1)        #Might also want to drop TimeofDay, since contained in other features

                  
#Noticed that the appeared Hour/Minute/Day/Month/Year weren't consisted with appearedLocalTime. Removed them all in favor of appearedLocalTime
train = train.drop(['appearedHour', 'appearedMinute', 'appearedDay', 'appearedMonth', 'appearedYear'],1)
#Convert appearedLocalTime string to DateTime
train['appearedLocalTime'] =  pd.to_datetime(train['appearedLocalTime'], format='%Y-%m-%dT%H:%M:%S')        #Note that %y is a 2digit, while %Y is 4digits for the year
#Now reinstate the appeared Hour/Minute/Day/Month/Year, then drop appearedLocalTime
train['appearedHour'] = train['appearedLocalTime'].dt.hour
train['appearedMinute'] = train['appearedLocalTime'].dt.minute
train['appearedDay'] = train['appearedLocalTime'].dt.day
train['appearedMonth'] = train['appearedLocalTime'].dt.month
train['appearedYear'] = train['appearedLocalTime'].dt.year
train = train.drop(['appearedLocalTime'],1)
#Now use 1-of-K encoding using pd.get_dummies()
Hour = pd.get_dummies(train.appearedHour, drop_first=True, prefix='hour')
Minute = pd.get_dummies(train.appearedMinute, drop_first=True, prefix='minute')
Day = pd.get_dummies(train.appearedDay, drop_first=True, prefix='day')
Month = pd.get_dummies(train.appearedMonth, drop_first=True, prefix='month')
Year = pd.get_dummies(train.appearedYear, drop_first=True, prefix='year')
train = train.join(Hour)         #To avoid dummy variable trap
train = train.join(Minute)
train = train.join(Day)
train = train.join(Month)
train = train.join(Year)
#Now we drop the appearedTimeX feature
train = train.drop(['appearedHour', 'appearedMinute', 'appearedDay', 'appearedMonth', 'appearedYear'],1)
                  

#Converting appearedTimeofDay into ordinal
time_mapping = {"morning": 0, "afternoon": 1, "evening": 2, "night": 3}
train['appearedTimeOfDay'] = train['appearedTimeOfDay'].map(time_mapping)


#Same for terrainType
Terrain = pd.get_dummies(train.terrainType, drop_first=True, prefix='terrain')
train = train.join(Terrain)         #To avoid dummy variable trap
#Now we drop the terrain feature
train = train.drop(['terrainType'],1)


#Get dummies on cities
City = pd.get_dummies(train.city, drop_first=True, prefix='city')
train = train.join(City)         #To avoid dummy variable trap
#Now we drop the city feature
train = train.drop(['city'],1)


#redefining continents such that they correspond to the main 7 continents
train.continent[train['continent']=='America/Indiana']='America'
train.continent[train['continent']=='America/Kentucky']='America'
train.continent[train['continent']=='Pacific']='Australia'
train.continent[train['continent']=='Atlantic']='Europe'
train.continent[train['continent']=='America/Argentina']='CentralAmerica'
#Then change them to dummies
Continent = pd.get_dummies(train.continent, drop_first=True, prefix='continent')
train = train.join(Continent)         #To avoid dummy variable trap
#Now we drop the continent feature
train = train.drop(['continent'],1)


#Comparing weather columns and choosing to drop weatherIcon. Then use dummies for weather
train['weather'].value_counts()
train['weatherIcon'].value_counts()             #These weather icons are based on time of day as well, making me inclined to not use them.
Weather = pd.get_dummies(train.weather, drop_first=True, prefix='weather')
train = train.join(Weather)         #To avoid dummy variable trap
#Now we drop both weather features
train = train.drop(['weatherIcon', 'weather'],1)


#Want to band windBearing into the 8 cardinal directions. (Probably used azimuth degrees where blowing north is 0 degrees and blowing west is 90 degrees)
#We define North as 0, NW as 1, W as 2, etc...
train.loc[(train['windBearing'] >= 337.5), 'windBearing'] = 0
train.loc[(train['windBearing'] < 22.5), 'windBearing'] = 0
train.loc[(train['windBearing'] >= 22.5) & (train['windBearing'] < 67.5), 'windBearing'] = 1
train.loc[(train['windBearing'] >= 67.5) & (train['windBearing'] < 112.5), 'windBearing'] = 2
train.loc[(train['windBearing'] >= 112.5) & (train['windBearing'] < 157.5), 'windBearing'] = 3
train.loc[(train['windBearing'] >= 157.5) & (train['windBearing'] < 202.5), 'windBearing'] = 4
train.loc[(train['windBearing'] >= 202.5) & (train['windBearing'] < 247.5), 'windBearing'] = 5
train.loc[(train['windBearing'] >= 247.5) & (train['windBearing'] < 292.5), 'windBearing'] = 6
train.loc[(train['windBearing'] >= 292.5) & (train['windBearing'] < 337.5), 'windBearing'] = 7
#Now make them into dummies
WindBearing = pd.get_dummies(train.windBearing, drop_first=True, prefix='windBearing')
train = train.join(WindBearing)         #To avoid dummy variable trap
#Now we drop the wind direction feature
train = train.drop(['windBearing'],1)



#Some quick functions for converting minutes for sunrise/sunset minute standardization
def OnlyPositiveTime(x):
    if x<0:
        return x+1440                   #Where 1440 = minutes per day
    else:
        return x
    
def OnlyNegativeTime(x):
    if x>0:
        return x-1440                   #Where 1440 = minutes per day
    else:
        return x
    
#Turned Sunrise/set Hour & Minute into dummies. Made sure that minutes since midnight for sunrise/set is positive (no negative minutes)
SunriseHour = pd.get_dummies(train.sunriseHour, drop_first=True, prefix='sunriseHour')
SunriseMinute = pd.get_dummies(train.sunriseMinute, drop_first=True, prefix='sunriseMinute')
SunsetHour = pd.get_dummies(train.sunsetHour, drop_first=True, prefix='sunsetHour')
SunsetMinute = pd.get_dummies(train.sunsetMinute, drop_first=True, prefix='sunsetMinute')
train = train.join(SunriseHour)         #To avoid dummy variable trap
train = train.join(SunriseMinute)
train = train.join(SunsetHour)
train = train.join(SunsetMinute)
#Now we drop the sunrise/set time features
train = train.drop(['sunriseHour', 'sunriseMinute', 'sunsetHour', 'sunsetMinute'],1)
train['sunriseMinutesMidnight'].apply(OnlyPositiveTime)
train['sunsetMinutesMidnight'].apply(OnlyPositiveTime)
#Make sure that each sighting's minutes since sunrise (sunriseMinutesSince) is positive & that sunsetMinutesBefore is negative
train['sunriseMinutesSince'].apply(OnlyPositiveTime)
train['sunsetMinutesBefore'].apply(OnlyNegativeTime)


#Change urban-suburban-urban into numeric values. 0=urban, 1=midurban, 2=suburban, 3=rural
#Dropping suburban and midurban columns, since they dont seem to be accurate. A sighting can't be both urban, suburban, and midurban if they are partitioned bands of population density
#Instead banding to get the urban, suburban, midurban, rural categorization, then changing to ordinal
train = train.drop(['urban', 'suburban', 'midurban', 'rural'],1)
train.loc[train['population_density'] < 200, 'population_density'] = 0
train.loc[(train['population_density'] >= 200) & (train['population_density'] < 400), 'population_density'] = 1
train.loc[(train['population_density'] >= 400) & (train['population_density'] < 800), 'population_density'] = 2
train.loc[train['population_density'] > 800, 'population_density'] = 3
#Just changing the name to show that I processed
train.rename(columns={'population_density' : 'Urbanity'}, inplace = True)


#Changing pokestopDistanceKm from a str to a float
PokestopDistance = pd.to_numeric(train['pokestopDistanceKm'], errors='coerce')
temporary = pd.concat([train, PokestopDistance], axis=1)
train = temporary.dropna()                  #This ends up dropping 39 instances. I'll find out what is causing the NaN's later (Though errors='coerce' made them NaN's)


#Making sure that pokemonID (the first column)) and class (the last column) are the same
row_ids = train[train['class'] != train.pokemonId].index        #This yields an empty set --> identical columns
#So now drop one of them and keep the other (for now) to use as the labels
train.drop(['class'],1)
#Note that train['pokemonId'].unique().shape[0] gives the # of unique pokemon ID's.     (which is 144)

#************************************************************************************************************

#Split data set (train) into training and validation sets.
train_features = train.drop(['pokemonId'],1)
train_labels = train['pokemonId']
X_train, X_test, Y_train, Y_test = train_test_split(train_features, train_labels, train_size = 0.7, random_state = 46)
X_train.shape, Y_train.shape, X_test.shape         #Run this to make sure the sizes are as expected.

#*****************************************************************************************************
#Trying out a nerual network to solve this. Network is (528 --> 264 --> 1 nodes)

#Loading data since Keras can only use numpy arrays
X_training = X_train.as_matrix()
Y_training = Y_train.values
X_testing = X_test.as_matrix()
Y_testing = Y_test.as_matrix()
X_training.shape, Y_training.shape, X_testing.shape, Y_testing.shape
#Defining Model
model = Sequential()
model.add(Dense(264, input_dim=528, init='uniform', activation='relu'))     
model.add(Dense(1, init='uniform', activation='sigmoid'))
#Compiling
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#Fitting to model
model.fit(X_training, Y_training, nb_epoch=10, batch_size=10, validation_split=0.2)
#Evaluating model
scores = model.evaluate(X_testing, Y_testing)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

#Not getting very good results. 0.5% accuracy is pretty bad.
#Maybe try to optimize. Maybe not a good problem to apply neural networks to?
#Not sure how to know whether to use or not yet. Will have to learn more.




'''                                 #Add on addiitonal metrics of precision and recall? F1 score? Since some of the pokemon are only rarely sighted? train['pokemonId'].value_counts() shows that pokemon 83 was seen once, for example
model = BernoulliNB()
model.fit(X_train, Y_train)
Y_pred = model.predict(X_test)
acc_1 = round(accuracy_score(Y_test, Y_pred)*100, 2)          #16.15% accuracy

model = linear_model.LogisticRegression()
model.fit(X_train, Y_train)
Y_pred = model.predict(X_test)
acc_2 = round(accuracy_score(Y_test, Y_pred)*100, 2)

model = SVC()
model.fit(X_train, Y_train)
Y_pred = model.predict(X_test)
acc_3 = round(accuracy_score(Y_test, Y_pred)*100, 2)

model = KNeighborsClassifier(n_neighbors = 3)           #How to know what # of neighbors to use? I think this should be 150 or however many pokemon there are?
model.fit(X_train, Y_train)                                     #66.13% accuracy
Y_pred = model.predict(X_test)
acc_4 = round(accuracy_score(Y_test, Y_pred)*100, 2)            #34.13% accuracy with 150 classes.

model = GaussianNB()
model.fit(X_train, Y_train)
Y_pred = model.predict(X_test)
acc_5 = round(accuracy_score(Y_test, Y_pred)*100, 2)            #99.37% accuracy

model = Perceptron()                                            #13.51% accuracy
model.fit(X_train, Y_train)
Y_pred = model.predict(X_test)
acc_6 = round(accuracy_score(Y_test, Y_pred)*100, 2)

model = LinearSVC()
model.fit(X_train, Y_train)
Y_pred = model.predict(X_test)
acc_7 = round(accuracy_score(Y_test, Y_pred)*100, 2)

model = SGDClassifier()
model.fit(X_train, Y_train)
Y_pred = model.predict(X_test)
acc_8 = round(accuracy_score(Y_test, Y_pred)*100, 2)            #14.46% accuracy

model = DecisionTreeClassifier()
model.fit(X_train, Y_train)
Y_pred = model.predict(X_test)
acc_9 = round(accuracy_score(Y_test, Y_pred)*100, 2)            #99.99% accuracy

model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, Y_train)
Y_pred = model.predict(X_test)
acc_10 = round(accuracy_score(Y_test, Y_pred)*100, 2)

models = pd.DataFrame({
    'Model' : ['BernoulliNB', 'SVC', 'KNeighbors', 'Gaussian', 'Perceptron', 'Linear SVC', 'Stochastic Gradient Decent', 'Decision Tree', 'Random Forest'],
    'Accuracy Score' : [acc_1, acc_3, acc_4, acc_5, acc_6, acc_7, acc_8, acc_9, acc_10]
    })  #Try adding in logistic regression much later (since it takes so long to run)
models.sort_values(by='Score', ascending=False)


       '''   




'''
       #This is also a way to run the learning algorithms, particularly if we're going for log_loss
model = BernoulliNB()
model.fit(X_train, Y_train)
predicted = np.array(model.predict_proba(X_test))
log_loss(Y_test, predicted) 
 
#Logistic Regression for comparison
model = LogisticRegression(C=.01)
model.fit(training[features], training['crime'])
predicted = np.array(model.predict_proba(validation[features]))
log_loss(validation['crime'], predicted) 
          '''
          
          
          
              
          
          
