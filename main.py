import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
#######################################################################################
# loading data and checking the stats of data
#######################################################################################
os.chdir('D:\\Projects\\HousePricePrediction')
# loading the dataset in dat variable
dat = pd.read_csv('kc_house_data.csv')
dat.head()
dat.describe()
#######################################################################################
# removing null values if any
#######################################################################################
dat = dat.dropna()
#######################################################################################
# checking the most common type of house according to bedroom
#######################################################################################
dat['bedrooms'].value_counts().plot(kind='bar')
plt.title('no. of bedrooms')
plt.xlabel('bedroom')
plt.ylabel('Count')
sns.despine()

plt.show()
#######################################################################################
# Visualizing the location of house according to the respective location
#######################################################################################
plt.figure(figsize=(10, 10))
sns.jointplot(x=dat.lat.values, y=dat.long.values, size=12)
plt.xlabel('Latitude')
plt.ylabel('Longitude')
plt.show()
sns.despine()
#######################################################################################
# checking how price is affected by sqft_living
#######################################################################################
plt.scatter(dat.price, dat.sqft_living)
plt.title('price against sqft')
plt.xlabel('sqft_living')
plt.ylabel('price')
plt.show()
#######################################################################################
# checking how location affects the price
#######################################################################################
plt.scatter(dat.price, dat.long)
plt.xlabel('location')
plt.ylabel('price')
plt.title('price vs location of the area')

plt.scatter(dat.price, dat.lat)
plt.xlabel("Price")
plt.ylabel('Latitude')
plt.title("Latitude vs Price")
#######################################################################################
# scatter plot for bedroom vs price
#######################################################################################
plt.scatter(dat.bedrooms, dat.price)
plt.title("Bedroom and Price ")
plt.xlabel("Bedrooms")
plt.ylabel("Price")
plt.show()
sns.despine
plt.scatter((dat['sqft_living'] + dat['sqft_basement']), dat['price'])
plt.xlabel('sqft_living + basement')
plt.ylabel('price')
plt.show()
plt.scatter(dat.waterfront, dat.price)
plt.title("Waterfront vs Price ( 0= no waterfront)")
plt.show()
plt.scatter(dat['yr_built'] + dat['bedrooms'], dat.price)
plt.xlabel('yr_built')
plt.ylabel('price')
plt.show()
train1 = dat.drop(['id', 'price'], axis=1)
train1.head()

# scatter plot for price and floors
dat.floors.value_counts().plot(kind='bar')
plt.show()
# scatter plot for price and floors
plt.scatter(dat.floors, dat.price)
plt.xlabel('floors')
plt.ylabel('price')
plt.show()
# scatter plot according to condition and price
plt.scatter(dat.condition, dat.price)
plt.xlabel('condition')
plt.ylabel('price')
plt.show()
# scatter plot for price and zipcode
plt.scatter(dat.zipcode, dat.price)
plt.title("Which is the pricey location by zipcode?")
plt.show()
# scatter plot for bedrooms vs bathrooms
plt.scatter(dat.bathrooms, dat.bedrooms)
plt.title('bedrooms and bathrooms')
plt.show()
# we can see in the scatter plot that there is one outlier in bedrooms which has 33 bedroom
#######################################################################################
## Checking the boxplot for

# bedrooms
# bathrooms
# sqft_living
# sqft_lot
# sqft_above
# sqft_basement
#######################################################################################
plt.boxplot(dat['bedrooms'])
plt.show()
# dat['bedrooms'].to
dat = dat.drop(15870, )
dat1 = dat[dat['bedrooms'] > 30]
print(dat1)
################################################################################
###### checking and removing outliers for price ################
################################################################################
plt.boxplot(dat['price'])
plt.title('price boxplot')
plt.show()
dat1 = dat[dat['price'] > 6000000]
print(dat1)
dat = dat.drop([3914, 7252, 9254], )
print(dat)
dat1 = dat[dat['price'] > 6000000]
print(dat1)
dat1 = dat[dat['price'] <= 0]
print(dat1)

################################################################################
###### checking and removing outliers for bathrooms
################################################################################
plt.boxplot(dat['bathrooms'])
plt.title('bathrooms boxplot')
plt.show()
dat1 = dat[dat['bathrooms'] <= 0]
print(dat1)
# dat1 = dat[dat['price'] > 8 ]
# print(dat1)
dat = dat.drop([875, 1149, 3119, 5832, 6994, 9773, 9854, 10481, 14423, 19452], )
print(dat)
dat1 = dat[dat['bathrooms'] <= 0]
print(dat1)
################################################################################
###### checking and removing outliers for sqft_living ################
################################################################################
plt.boxplot(dat['sqft_living'])
plt.title('sqft_living boxplot')
plt.show()
dat1 = dat[dat['sqft_living'] > 10000]
print(dat1)
dat = dat.drop([12777], )
print(dat)
dat1 = dat[dat['sqft_living'] > 10000]
print(dat1)
################################################################################
###### checking and removing outliers for sqft_lot ################
################################################################################

plt.boxplot(dat['sqft_lot'])
plt.title('sqft_lot boxplot')
plt.show()
dat1 = dat[dat['sqft_lot'] > 1000000]
print(dat1)
dat = dat.drop([1719, 7647, 7769, 17319], )
print(dat)
dat1 = dat[dat['sqft_lot'] > 1000000]
print(dat1)
################################################################################
###### checking and removing outliers for sqft_above ################
################################################################################
plt.boxplot(dat['sqft_above'])
plt.title('sqft_above boxplot')
plt.show()
dat1 = dat[dat['sqft_above'] > 7000]
print(dat1)
# dat = dat.drop([12777], )
# print(dat)
# dat1 = dat[dat['sqft_above']> 10000]
# print(dat1)
################################################################################
###### checking and removing outliers for sqft_basement ################
################################################################################
plt.boxplot(dat['sqft_basement'])
plt.title('sqft_basement boxplot')
plt.show()

dat1 = dat[dat['sqft_basement'] > 4000]
print(dat1)
dat = dat.drop([8092], )
print(dat)
dat1 = dat[dat['sqft_basement'] > 4000]
print(dat1)
#######################################################################################
# checking the corelation matrix
#######################################################################################
corelationPlot = dat.corr()
print(corelationPlot)
#######################################################################################
# removing sqft living15 as it is highly corelated with sqft_living
#######################################################################################
dat = dat.drop('sqft_living15', axis=1)
corelationPlot = dat.corr()
print(corelationPlot)
#######################################################################################
# removing grade as it is highly corelated with sqft_living15
#######################################################################################
dat = dat.drop('grade', axis=1)
corelationPlot = dat.corr()
print(corelationPlot)
#######################################################################################
# removing sqft_lot15 as it is highly corelated with sqft_lot
#######################################################################################
dat = dat.drop('sqft_lot15', axis=1)
corelationPlot = dat.corr()

print(corelationPlot)
#######################################################################################
# Multiple Linear regression
#######################################################################################
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
labels = dat['price']
conv_dates = [1 if values == 2014 else 0 for values in dat.date]
dat['date'] = conv_dates
train1 = dat.drop(['id', 'price'], axis=1)
# Splitting the data in test and train
from sklearn.model_selection import train_test_split
# train data is set to 90% and 30% of the data to be my test data
x_train, x_test, y_train, y_test = train_test_split(train1, labels, test_size=0.30, random_state=2)
reg.fit(x_train, y_train)
reg.score(x_test, y_test)
#######################################################################################
# Grading boosting regression
#######################################################################################
from sklearn import ensemble
clf = ensemble.GradientBoostingRegressor(n_estimators=400, max_depth=5, min_samples_split=2,
 learning_rate=0.1, loss='ls')

clf.fit(x_train, y_train)
clf.score(x_test, y_test)
#######################################################################################
# Random forest
#######################################################################################
# Import the model we are using
from sklearn.ensemble import RandomForestRegressor
# Instantiate model with 1000 decision trees
rf = RandomForestRegressor(n_estimators=500, random_state=42)
# Train the model on training data
rf.fit(x_train, y_train);
# Use the forest's predict method on the test data
predictions = rf.predict(x_test)
# Calculate the absolute errors
errors = abs(predictions - y_test)
# Print out the mean absolute error (mae)
print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')
# Calculate mean absolute percentage error (MAPE)
mape = 100 * (errors / y_test)
# Calculate and display accuracy
accuracy = 100 - np.mean(mape)
print('Accuracy:', round(accuracy, 2), '%.')
#######################################################################################
# XGrading boosting regression
#######################################################################################

from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

model = XGBClassifier()
model.fit(x_train, y_train)
print(model)
# Xgboost took too much time to execute
# make predictions for test data
y_pred = model.predict(x_test)
predictions = [round(value) for value in y_pred]

# evaluate predictions
accuracyXgBoost = accuracy_score(y_test, predictions)
print("Accuracy: xgboost %.2f%%" % (accuracyXgBoost * 100.0))