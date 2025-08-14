# 1️⃣ One-Line Summary
# This code trains a Logistic Regression model using sonar signal data to classify whether an object is a mine or a rock, then tests its accuracy and predicts a sample input.

# 2️⃣ Line-by-Line Explanation

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
# Importing libraries:
# numpy → Handles numerical arrays
# pandas → For reading and working with CSV data
# train_test_split → Splits dataset into training and testing sets
# LogisticRegression → ML model for classification
# accuracy_score → Measures model accuracy

sonar_data = pd.read_csv('/content/sonar.csv', header=None)
# Reads the sonar dataset into a Pandas DataFrame
# header=None means no column names, so columns are numbered 0–60

sonar_data.head()
# Displays the first 5 rows of the dataset

sonar_data.shape
# Shows the number of rows and columns in the dataset

sonar_data.describe()
# Displays statistical details (mean, std, min, max) for each column

sonar_data[60].value_counts()
# Counts how many rocks (R) and mines (M) exist in the dataset

sonar_data.groupby(60).mean()
# Groups rows by label (R/M) and calculates mean for each column

X = sonar_data.drop(columns=60, axis=1)
Y = sonar_data[60]
# Separates features (X) from labels (Y)
# X = all columns except label
# Y = label column (R or M)

print(X)
print(Y)
# Prints features and labels

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.1, stratify=Y, random_state=1
)
# Splits the dataset into 90% training and 10% testing
# stratify=Y keeps class balance
# random_state=1 ensures reproducible results

print(X.shape, X_train.shape)
# Prints size of full dataset and training set

model = LogisticRegression()
# Creates a Logistic Regression model instance

model.fit(X_train, Y_train)
# Trains the model using training data

X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print(training_data_accuracy)
# Predicts on training data
# Compares predictions to true labels for training accuracy
# Prints training accuracy

X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print(test_data_accuracy)
# Predicts on test data
# Compares predictions to true labels for test accuracy
# Prints test accuracy

input_data = (0.0333,0.0221,0.0270,0.0481,0.0679,0.0981,0.0843,0.1172,0.0759,0.0920,
              0.1475,0.0522,0.1119,0.0970,0.1174,0.1678,0.1642,0.1205,0.0494,0.1544,
              0.3485,0.6146,0.9146,0.9364,0.8677,0.8772,0.8553,0.8833,1.0000,0.8296,
              0.6601,0.5499,0.5716,0.6859,0.6825,0.5142,0.2750,0.1358,0.1551,0.2646,
              0.1994,0.1883,0.2746,0.1651,0.0575,0.0695,0.0598,0.0456,0.0021,0.0068,
              0.0036,0.0022,0.0032,0.0060,0.0054,0.0063,0.0143,0.0132,0.0051,0.0041)
# Example sonar reading for prediction

input_data_as_numpy_array = np.asarray(input_data)
# Converts the input data to a numpy array

input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
# Reshapes it so the model sees it as a single row with multiple features

prediction = model.predict(input_data_reshaped)
print(prediction)
# Predicts class (R or M) for the given input and prints the result
