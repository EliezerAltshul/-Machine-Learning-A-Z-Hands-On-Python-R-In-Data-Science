##Import Libraries

#numpy will allow to work with arrays
#matplotlib is for plats
#panda imports data and matrices
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot
import pandas as pd

#Import dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

#Splitting the dataset into the Training set and the Testing set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test \
    = train_test_split(X, y, test_size=0.2, random_state=1)

#Training the Simple Linear Regression model on the Training Set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#Predicting the Test set results
y_pred = regressor.predict(X_test)

#Visualizing the Traing set results
plt.scatter(X_train, y_train, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title('Salary vs Experience (Training Set)')
plt.xlabel('Years of Experience')
plt.xlabel('Salary')
plt.show()

#Visualizing the Test set result
plt.scatter(X_test, y_test, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title('Salary vs Experience (Test Set)')
plt.xlabel('Years of Experience')
plt.xlabel('Salary')
plt.show()