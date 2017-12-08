import numpy as np 
import random

def sigmoid(x):
	return 1/(1+np.exp(-x))

def sigmoid_der(x):
	return x*(1-x)



class Multilayer_Perceptron:
	"""docstring for Multilayer_Percetron"""
	def __init__(self, data):
		# extracting x1 and x2 from data points 
		self.X = data[:,:-1]
		# extracting the target value of i.e, from data 
		self.Y = data[:,-1].reshape(data.shape[0],1)
		# no of units in hidden layer
		self.a = 8
		# initializing the weights for input layer
		self.wi = np.random.random((2,self.a))
		# initializing the weights for hidden layer
		self.wh = np.random.random((self.a, 1))

		# print("X is :", self.X.shape[1], "and Y is : ", self.Y)
	
	def predict(self, inputs):
		# inputs contain only (x1 , x2) on which we will predict our probabilities of output
		s1 = sigmoid(np.dot(inputs, self.wi))
		s2 = sigmoid(np.dot(s1, self.wh))
		return s2


	def train(self, batch = 2, epochs=1, learn_rate = 0.5):
		for i in range(epochs):
			#iterating over the batches
			for i in range(0, self.X.shape[0], batch):
				# getting the batches (x1,x2)
				batch_in = self.X[i:i+batch]
				# getting the target output (y) of the batch
				batch_targ_out = self.Y[i:i+batch].reshape(batch,1)
				#*******feed forward start*********#
				s1 = sigmoid(np.dot(batch_in, self.wi))
				s2 = sigmoid(np.dot(s1, self.wh))
				#*******feed forward finished ******#

				#*********back propogation starts******#
				l2_err = batch_targ_out - s2
				l2_delta = np.multiply(l2_err, sigmoid_der(s2))

				l1_err = np.dot(l2_delta, self.wh.T)
				l1_delta = np.multiply(l1_err, sigmoid_der(s1))

				self.wh += learn_rate*np.dot(s1.T, l2_delta)
				self.wi += learn_rate*np.dot(batch_in.T, l1_delta)
				#********back propogation finish *********#



# data format ( x1, x2, y)
data = np.array([[0,0,1], [0,1,0],[1,0,0],[1,1,1],[2,2,0],[3,4,1]])

n = Multilayer_Perceptron(data)
# please put the batch size  as only factors of len(data)
n.train(epochs = 100000, batch=3, learn_rate=0.2)
inputs = data[:6,:-1]
# inputs contains only the (x1, x2) format 
print("after training : ", n.predict(inputs))
