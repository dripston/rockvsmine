# rockvsmine
This imports the required libraries for data handling, model training, and evaluation.
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

This loads the sonar dataset into a Pandas DataFrame without headers.
sonar_data = pd.read_csv('/content/sonar.csv', header=None)

This displays the first 5 rows of the dataset.
sonar_data.head()

This shows the number of rows and columns in the dataset.
sonar_data.shape

This gives statistical details like mean, standard deviation, and range for each feature.
sonar_data.describe()

This counts how many samples belong to each class (R or M).
sonar_data[60].value_counts()

This groups the dataset by label and calculates the mean for each feature.
sonar_data.groupby(60).mean()

This separates the features (X) from the labels (Y).
X = sonar_data.drop(columns=60, axis=1)
Y = sonar_data[60]

This prints the feature set and label set.
print(X)
print(Y)

This splits the data into 90% training and 10% testing sets while keeping class balance.
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, stratify=Y, random_state=1)

This prints the shape of the dataset before and after splitting.
print(X.shape, X_train.shape)

This creates a Logistic Regression model instance.
model = LogisticRegression()

This trains the Logistic Regression model with the training data.
model.fit(X_train, Y_train)

This predicts labels for the training data.
X_train_prediction = model.predict(X_train)

This calculates the accuracy of predictions on the training data.
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)

This prints the training accuracy score.
print(training_data_accuracy)

This predicts labels for the test data.
X_test_prediction = model.predict(X_test)

This calculates the accuracy of predictions on the test data.
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)

This prints the test accuracy score.
print(test_data_accuracy)

This is a sample input representing sonar readings for prediction.
input_data = (0.0333,0.0221,0.0270,0.0481,0.0679,0.0981,0.0843,0.1172,0.0759,0.0920,
              0.1475,0.0522,0.1119,0.0970,0.1174,0.1678,0.1642,0.1205,0.0494,0.1544,
              0.3485,0.6146,0.9146,0.9364,0.8677,0.8772,0.8553,0.8833,1.0000,0.8296,
              0.6601,0.5499,0.5716,0.6859,0.6825,0.5142,0.2750,0.1358,0.1551,0.2646,
              0.1994,0.1883,0.2746,0.1651,0.0575,0.0695,0.0598,0.0456,0.0021,0.0068,
              0.0036,0.0022,0.0032,0.0060,0.0054,0.0063,0.0143,0.0132,0.0051,0.0041)

This converts the input data to a NumPy array.
input_data_as_numpy_array = np.asarray(input_data)

This reshapes the array for a single sample prediction.
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

This predicts the label (R or M) for the given input data.
prediction = model.predict(input_data_reshaped)

This prints the prediction result.
print(prediction)
