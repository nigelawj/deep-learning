# Self Organizing Map

# Part 1 - SOM

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Credit_Card_Applications.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler(feature_range=(0, 1))
X = sc.fit_transform(X)

# Training the SOM
from minisom import MiniSom

# Plot a 10x10 SOM
som = MiniSom(x=10, y=10, input_len=15, sigma=1.0, learning_rate=0.5)
som.random_weights_init(X)
som.train_random(data=X, num_iteration=100)

# Visualizing the results
from pylab import bone, pcolor, colorbar, plot, show

bone()
pcolor(som.distance_map().T)
colorbar()

markers = ['o', 's']
colors = ['r', 'g']

for i, x in enumerate(X):
    # winning neuron/node is the neuron that the customer is mapped to (closest neuron)
    w = som.winner(x)
    # place marker in the middle
    plot(w[0] + 0.5,
         w[1] + 0.5,
         markers[y[i]],
         markeredgecolor=colors[y[i]],
         markerfacecolor='None',
         markersize=10,
         markeredgewidth=2)
show()

# Finding the frauds
mappings = som.win_map(X) # not every node has a customer matched to it

# This line must be changed every run as mappings change every run
frauds = np.concatenate((mappings[(8,5)], mappings[(8,6)], mappings[(8,7)]), axis=0)
frauds = sc.inverse_transform(frauds)

# nodes with customers that have a high Mean Interneuron Distance (MID) have outliers in them; higher chance of them being outliers since after the long process of the SOM which will group up customers with similar attributes they still 'stand out'

# Part 2 - Going from Unsupervised to Supervised Deep Learning

# Creating the matrix of features
customers = dataset.iloc[:, 1:].values # ditch the customer ID

# Creating the dependent variable (create a is_fraud attribute: the dependent y-variable)
is_fraud = np.zeros(len(dataset))

for i in range(len(dataset)):
    if dataset.iloc[i,0] in frauds:
        is_fraud[i] = 1

# Feature Scaling
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
customers = sc.fit_transform(customers)

# Importing the Keras libraries and packages
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(units=2, kernel_initializer='uniform', activation='relu', input_dim=15))

# Adding the output layer
classifier.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))

# Compiling the ANN
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(customers, is_fraud, batch_size=1, epochs=2)

# Predicting the probabilities of frauds
y_pred = classifier.predict(customers)

y_pred = np.concatenate((dataset.iloc[:, 0:1].values, y_pred), axis=1)
y_pred = y_pred[y_pred[:, 1].argsort()] # y_pred contains customers and their predicted probabilities of being frauds, in descending order