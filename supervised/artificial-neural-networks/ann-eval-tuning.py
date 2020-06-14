# Artificial Neural Network

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

# Part 2 - Evaluating, Improving and Tuning the ANN

# Evaluating the ANN
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import ModelCheckpoint

def build_classifier(): # Function to build the ANN
    classifier = Sequential()
    classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu', input_dim=11))
    classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu'))
    classifier.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))
    classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return classifier

classifier = KerasClassifier(build_fn=build_classifier, batch_size=10, epochs=100)

accuracies = cross_val_score(estimator=classifier, X=X_train, y=y_train, cv=10, n_jobs=-1) # change this if code crashes... n_jobs=1 or 2 is a safe (slow) choice
mean = accuracies.mean()
variance = accuracies.std()

# Improving the ANN
# Dropout Regularization to reduce overfitting if needed

# Tuning the ANN
from sklearn.model_selection import GridSearchCV

def build_classifier2(optimizer):
    classifier = Sequential()
    classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu', input_dim=11))
    classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu'))
    classifier.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))
    classifier.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    return classifier

classifier2 = KerasClassifier(build_fn=build_classifier2)

parameters = {
    'batch_size': [25, 32],
    'epochs': [100, 500],
    'optimizer': ['adam', 'rmsprop'],
}

grid_search = GridSearchCV(estimator=classifier2,
                           param_grid=parameters,
                           scoring='accuracy',
                           cv=10, n_jobs=-1) # change this if code crashes... n_jobs=1 or 2 is a safe (slow) choice

grid_search = grid_search.fit(X_train, y_train)

best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_