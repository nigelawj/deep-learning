# Stacked Autoencoders

# Importing the libraries
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable

# Importing the dataset
movies = pd.read_csv('ml-1m/movies.dat', sep='::', header=None, engine='python', encoding='latin-1')
users = pd.read_csv('ml-1m/users.dat', sep='::', header=None, engine='python', encoding='latin-1')
ratings = pd.read_csv('ml-1m/ratings.dat', sep='::', header=None, engine='python', encoding='latin-1')

# Preparing the training set and the test set
# Take only 1 train_test_split (u1)
training_set = pd.read_csv('ml-100k/u1.base', delimiter='\t') # sep and delimiter are the same
training_set = np.array(training_set, dtype='int')

test_set = pd.read_csv('ml-100k/u1.test', delimiter='\t')
test_set = np.array(test_set, dtype='int')

# Getting the number of users and movies
num_users = int(max(max(training_set[:, 0]), max(test_set[:, 0]))) # find largest id in all the data
num_movies = int(max(max(training_set[:, 1]), max(test_set[:, 1])))

# Converting the data into an array with users in lines and movies in columns
def convert(data):
	new_data = []
	for idx in range(1, num_users + 1):
		movies_id = data[:, 1][data[:, 0] == idx]
		ratings_id = data[:, 2][data[:, 0] == idx]

		ratings = np.zeros(num_movies)
		ratings[movies_id - 1] = ratings_id

		new_data.append(list(ratings))
	
	return new_data

training_set = convert(training_set)
test_set = convert(test_set)

# Converting the data into Torch tensors
training_set = torch.FloatTensor(training_set)
test_set = torch.FloatTensor(test_set)

# Creating the architecture of the Neural Network
class SAE(nn.Module):
	def __init__(self, ):
		super(SAE, self).__init__() # inherit nn.Module
		# encode
		self.fc1 = nn.Linear(num_movies, 20) # 20 neurons; can be tuned
		self.fc2 = nn.Linear(20, 10)
		# decode
		self.fc3 = nn.Linear(10, 20)
		self.fc4 = nn.Linear(20, num_movies)
		self.activation = nn.Sigmoid()

	def forward(self, x):
		# encode
		x = self.activation(self.fc1(x))
		x = self.activation(self.fc2(x))
		# decode
		x = self.activation(self.fc3(x))
		x = self.fc4(x)

		return x

sae = SAE()
criterion = nn.MSELoss()
optimiser = optim.RMSprop(sae.parameters(), lr=0.01, weight_decay=0.5)

# Training the SAE
num_epochs = 200
for epoch in range(1, num_epochs+1):
	train_loss = 0
	s = 0. # no. of users that rated at least 1 movie

	for user_id in range(num_users):
		input = Variable(training_set[user_id]).unsqueeze(0)
		target = input.clone()

		if (torch.sum(target.data > 0) > 0): # contains at least 1 rating > 0
			output = sae(input)
			target.requires_grad = False
			output[target == 0] = 0 # ignore indexes where input was 0

			loss = criterion(output, target)
			# avg. error of movies with non-zero ratings
			mean_corrector = num_movies/(float(torch.sum(target.data > 0) + (1e-10))) # ensure denominator != 0
			loss.backward() # decides direction which weights will be updated (increased/decreased)

			train_loss += np.sqrt(loss.data*mean_corrector) # loss is error in stars; loss of 1 == prediction differs from actual data by 1 star of rating
			s += 1.
			optimiser.step() # decides intensity (amount) of weights update
		
	print(f'epoch: {epoch}\nloss: {train_loss/s}\n')

# Testing the SAE
test_loss = 0
s = 0.

for user_id in range(num_users):
	input = Variable(training_set[user_id]).unsqueeze(0)
	target = Variable(test_set[user_id]).unsqueeze(0)

	if (torch.sum(target.data > 0) > 0): # contains at least 1 rating > 0
		output = sae(input)
		target.requires_grad = False
		output[target == 0] = 0 # ignore indexes where input was 0

		loss = criterion(output, target)
		# avg. error of movies with non-zero ratings
		mean_corrector = num_movies/float(torch.sum(target.data > 0) + (1e-10)) # ensure denominator != 0

		test_loss += np.sqrt(loss.data*mean_corrector)
		s += 1.
	
print(f'test loss: {test_loss/s}\n') # achieved a loss of 0.951! - i.e. prediction error of < 1 star (within 1 star error margin)
