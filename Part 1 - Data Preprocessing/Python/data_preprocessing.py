##Import Libraries

#numpy will allow to work with arrays
#matplotlib is for plats
#panda imports data and matrices
import numpy as np
import matplotlib.pyplot
import pandas as pd

#Import dataset
customer_dataset = pd.read_csv('Data.csv')
X = customer_dataset.iloc[:, :-1].values
y = customer_dataset.iloc[:, -1].values

print(X)
print(y)

#Taking care of missing data
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])

print(X)
print(y)

#Encoding categorical data

#Encoding independent variables
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
X = np.array(ct.fit_transform(X))
print(X)

#Encoding dependent variables
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)
print(y)

#Splitting the dataset into the Training set and the Testing set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test \
    = train_test_split(X, y, test_size=0.2, random_state=1)
print("Customer Feature Train: ", X_train)
print("Customer Feature Test: ", X_test)
print("Customer Purchase Train: ", y_train)
print("Customer Purchase Test: ", y_test)

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train[:, 3:] = sc.fit_transform(X_train[:, 3:])
X_test[:, 3:] = sc.transform(X_test[:, 3:])
print("Customer Feature Train: ", X_train)
print("Customer Feature Test: ", X_test)








