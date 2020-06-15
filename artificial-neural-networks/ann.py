# Artificial Neural Network - Predicting probabilities a customer leaves the bank

# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Encoding categorical data
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# Label Encode the 2 categorical features: Country and Gender
le_X1 = LabelEncoder()
X[:, 1] = le_X1.fit_transform(X[:, 1])
le_X2 = LabelEncoder()
X[:, 2] = le_X2.fit_transform(X[:, 2])

# One Hot Encode Country; Gender not needed since binary
# zz = onehotencoder.fit_transform(X[:, 1].reshape(-1, 1)).toarray() # reshape converts the column '1' into a 2D np.array:
# [1.0, 0.0, ...] reshaped into [[1.0], [0.0], ...]
# Outputs the OHEd 2D array; need to put it back into X
ohe_cat = [1] # List of categories to OneHotEncode: actually just the column '1'
ct = ColumnTransformer([('encoder', OneHotEncoder(), ohe_cat)], remainder='passthrough')
X = np.array(ct.fit_transform(X), dtype=np.float) # Not sure why must enforce np.float...

# Avoid dummy variable trap
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Part 2 - Creating the ANN

# Importing the Keras libraries and packages
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu', input_dim=11))

# Adding the second hidden layer
classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu'))

# Adding the output layer
classifier.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))

# Compiling the ANN
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size=10, epochs=100)

# Part 3 - Making predictions and evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Predicting a single new observation
"""
Predict if the customer with the following informations will leave the bank:
Geography: France
Credit Score: 600
Gender: Male
Age: 40
Tenure: 3
Balance: 60000
Number of Products: 2
Has Credit Card: Yes
Is Active Member: Yes
Estimated Salary: 50000
"""
new_prediction = classifier.predict(sc.transform(np.array([[0.0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])))
new_prediction = (new_prediction > 0.5)
print(f'Prediction: {new_prediction[0][0]}')