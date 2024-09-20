# Import essential Libraries
import numpy as np
import pandas as pd
import joblib
from joblib import dump, load

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# create Pandas DataFrame
rentalDF = pd.read_csv('data/rental_1000.csv')

# Consider Features and Labels
X = rentalDF[['rooms','sqft']].values   # Features
y = rentalDF['price'].values           # Labels

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.20)

# Training the Dataset
lr = LinearRegression()
model = lr.fit(X_train, y_train)

joblib.dump(model,'model/rentalprediciton.joblib')
