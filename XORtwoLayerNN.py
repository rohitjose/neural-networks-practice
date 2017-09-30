# Two Layer XOR Neural Network

import numpy as np
import time

# Activation function - Sigmoid Function
# Sigmoid function turns numbers to probabilities. In the NN the weights are a set of probabilities.
def sigmoid(x):
	return 1.0/(1.0 + np.exp(-x))

# Activation fucntion - Tanh Function
# We will be using two activation functions - tangent function is trains a more accurate model
def tanh_prime(x):
	return 1 - np.tanh(x) ** 2

# Training function
# x - input data
# t - transpose that helps with matrix multiplication
# V & W - are the training weights for the two layers
# bv & bw - are the biases 
def train(x, t, V, W, bv, bw):
	# Forward propagation - matrix multiplication + biases
	# Layer 1
	A = np.dot(x,V) + bv
	Z = np.tanh(A)
	# Layer 2
	B = np.dot(Z,W) + bw
	Y = sigmoid(B)

	# Back propagation
	Ew = Y - t
	Ev = tanh_prime(A) * np.dot(W, Ew) 

	# Predict the loss
	dW = np.outer(Z, Ew)
	dV = np.outer(x, Ev)

	# Cross entropy - Used because this is a classification task (improves accuracy)
	loss = -np.mean(t * np.log(Y) + (1 - t) * np.log(1 - Y))

	return loss, (dV, dW, Ev, Ew)

# Prediction function
# Predicts the values on a test set
# x - input data
# V & W - are the training weights for the two layers
# bv & bw - are the biases 
def predict(x, V, W, bv, bw):
	A = np.dot(x, V) + bv
	B = np.dot(np.tanh(A) , W) + bw
	return (sigmoid(B) > 0.5).astype(int)




# Variables
n_hidden = 10
n_in = 10
# Outputs
n_out = 10

# Sample data - to be generated
n_sample = 300

# Hyper Parameters
learning_rate = 0.01
momentum = 0.9

# Seed for random number generation
np.random.seed(1) # Non-deterministic Seeding (Generates the same random numbers every time the code runs)

# Create Layers
# Weights
V = np.random.normal(scale = 0.1, size = (n_in, n_hidden))
W = np.random.normal(scale = 0.1, size = (n_hidden, n_out))
# Biases - initialize biases
bv = np.zeros(n_hidden)
bw = np.zeros(n_out)

params = [V, W, bv, bw]

# Generate data
X = np.random.binomial(1, 0.5, (n_sample, n_in))
T = X ^ 1

# Training
for epoch in range(100):
	err = []
	update = [0] * len(params)

	t0 = time.clock()
	# Weight update for each data point
	for i in range(X.shape[0]):
		loss, gradient = train(X[i], T[i], *params)
		# Update loss
		for j in range(len(params)):
			params[j] -= update[j]

		for j in range(len(params)):
			update[j] = learning_rate * gradient[j] + momentum * update[j]

		err.append(loss)

	print("Epoch: %d, Loss: %.8f Time: %.4fs" % (epoch, np.mean(err), time.clock() - t0))

# Prediction
x = np.random.binomial(1, 0.5, n_in)
print('XOR Prediction')
print(x)
print(predict(x, *params))