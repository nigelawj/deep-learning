# Restricted Boltzmann Machines

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

# Converting the ratings into binary ratings 1 (Liked) or 0 (Not Liked)
training_set[training_set == 0] = -1
training_set[training_set == 1] = 0
training_set[training_set == 2] = 0
training_set[training_set >= 3] = 1

test_set[test_set == 0] = -1
test_set[test_set == 1] = 0
test_set[test_set == 2] = 0
test_set[test_set >= 3] = 1

# Creating the architecture of the Neural Network
class RBM():
	def __init__(self, nv, nh):
		"""
		nv: number of visible nodes (dataset, i.e. no. of movies)
		
		nh: number of hidden nodes (features, e.g. actor, movie length, whether movie is old/new, etc.)
		"""
		self.W = torch.randn(nh, nv) # initialise weights
		# initialise biases
		self.a = torch.randn(1, nh) # probability of visible nodes given the hidden nodes
		self.b = torch.randn(1, nv)

	def sample_h(self, x):
		wx = torch.mm(x, self.W.t())
		activation = wx + self.a.expand_as(wx) # apply biases to each line of a mini batch
		p_h_given_v = torch.sigmoid(activation)
		
		return p_h_given_v, torch.bernoulli(p_h_given_v)

	def sample_v(self, y):
		wy = torch.mm(y, self.W)
		activation = wy + self.b.expand_as(wy)
		p_v_given_h = torch.sigmoid(activation)
		
		return p_v_given_h, torch.bernoulli(p_v_given_h)

	def train(self, v0, vk, ph0, phk):
		'''
		v0: visible nodes (input vector containing ratings of all movies rated by a particular user)

		vk: visible nodes obtained after k iterations of k-step contrastive divergence
		
		ph0: vector of probabilities, at 1st iteration, that a hidden node == 1, given values of v0
		
		phk: probabilities of hidden nodes after k iterations given values of visible nodes vk
		'''
		self.W += (torch.mm(v0.t(), ph0) - torch.mm(vk.t(), phk)).t()
		self.b += torch.sum((v0 - vk), 0)
		self.a += torch.sum((ph0 - phk), 0)

nv = len(training_set[0])
nh = 100
batch_size = 100

rbm = RBM(nv, nh)

# Training the RBM
num_epochs = 10

for epoch in range(1, num_epochs + 1):
	train_loss = 0
	s = 0. # initialise as float
	
	for user_id in range(0, num_users - batch_size, batch_size):
		vk = training_set[user_id:user_id+batch_size]
		v0 = training_set[user_id:user_id+batch_size]
		ph0,_ = rbm.sample_h(v0)

		for k in range(10):
			_,hk = rbm.sample_h(vk)
			_,vk = rbm.sample_v(hk)
			vk[v0<0] = v0[v0<0]

		phk,_ = rbm.sample_h(vk)
		rbm.train(v0, vk, ph0, phk)

		train_loss += torch.mean(torch.abs(v0[v0>=0] - vk[v0>=0]))
		s += 1.

	print(f'epoch: {epoch}\nloss: {train_loss/s}')

# Testing the RBM
test_loss = 0
s = 0. # initialise as float

for user_id in range(0, num_users):
	v = training_set[user_id:user_id+1]
	vt = test_set[user_id:user_id+1]

	if (len(vt[vt>=0]) > 0):
		_,h = rbm.sample_h(v)
		_,v = rbm.sample_v(h)

		test_loss += torch.mean(torch.abs(vt[vt>=0] - v[vt>=0]))
		s += 1.

print(f'test loss: {test_loss/s}')