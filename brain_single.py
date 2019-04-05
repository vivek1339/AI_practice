import numpy as np
from numba import vectorize
import sys
import math
import pickle
import random
from math_support import loss,sigmoidprime,sigmoid
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable
from scipy.stats import logistic
import warnings
from multiprocessing import Pool
class network():
	def __init__(self):
		self.input=[]
		self.inputsize=9
		self.outputsize=9
		self.hidden1=27
		self.lr=0.01
		self.input_to_hidden1=np.random.randn(self.inputsize,self.hidden1)
		self.hidden1_to_hidden2=np.random.randn(self.hidden1,self.hidden1)
		self.hidden2_to_output=np.random.randn(self.hidden1,self.outputsize)
		self.output=np.zeros((self.outputsize,1),dtype=float)
		#warnings.filterwarnings("")
	def forward(self,x):
		self.output=[]
		self.input=x

		self.layer_1=np.dot(x,self.input_to_hidden1)
		self.layer_1=sigmoid(self.layer_1)
		self.layer_2=np.dot(self.layer_1,self.hidden1_to_hidden2)
		self.layer_2=sigmoid(self.layer_2)
		self.output=np.dot(self.layer_2,self.hidden2_to_output)
		self.output=sigmoid(self.output)
		return np.array(self.output)
	def back_propogation(self):
		try:
			#self.o_error = y - self.output
			self.o_delta=np.array(self.o_error)*sigmoidprime(self.output)
			self.layer_2_error=self.o_delta.dot(self.hidden2_to_output.T)
			self.layer_2_delta=np.array(self.layer_2_error)*sigmoidprime(self.layer_2)
			self.layer_1_error=self.layer_2_delta.dot(self.hidden1_to_hidden2.T)
			self.layer_1_delta=np.array(self.layer_1_error)*sigmoidprime(self.layer_1)
			self.input_to_hidden1 += np.dot(self.input.T,self.layer_1_delta)*self.lr
			self.hidden1_to_hidden2 +=self.layer_1.T.dot(self.layer_2_delta)*self.lr
			self.hidden2_to_output += self.layer_2.T.dot(self.o_delta)*self.lr
		except RuntimeWarning:
			print(sigmoidprime(self.layer_1_error))
			sys.exit(0)
	def train(self,x,y):
		#print(self.hidden1_to_output)
		#print("here change")
		o=self.forward(x)
		self.back_propogation(y)
		#print(self.hidden1_to_output)
class ReplayMemory(object):
    
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
    
    def push(self, event):
        self.memory.append(event)
        if len(self.memory) > self.capacity:
            del self.memory[0]
    
    def sample(self, batch_size):
        samples = zip(*random.sample(self.memory, batch_size))
        return map(lambda x: Variable(torch.cat(x, 0)), samples)
class dgn():
	def __init__(self, input_size, gamma):
		self.loss_array =[]
		self.gamma = gamma
		self.reward_window = []
		self.model=network()
		self.memory = ReplayMemory(100000)
		self.last_state = torch.Tensor(input_size).unsqueeze(0)
		self.last_action = 0
		self.last_reward = 0
		self.temprature=1
		#torch.set_default_tensor_type('torch.DoubleTensor')
	def select_action(self,state):
		probs=F.softmax(torch.tensor(self.model.forward(state),dtype=torch.float)*self.temprature,1)#temprature=10
		action=probs.multinomial(1)
		return action.data[0]
	def learn(self, batch_state, batch_next_state, batch_reward, batch_action):
		outputs=torch.tensor(self.model.forward(batch_state)).gather(1, batch_action.unsqueeze(1)).squeeze(1)
		next_output=torch.tensor(self.model.forward(batch_next_state)).detach().max(1)[0]
		target=self.gamma*next_output.float()+batch_reward.float()
		self.model.o_error = F.smooth_l1_loss(outputs, target.double())
		if(len(self.loss_array)>200):
			del self.loss_array[0]
		self.loss_array.append(self.model.o_error.sum())
		self.model.back_propogation()
	def update(self, reward, new_state):
		#print(new_state)
		new_state = torch.Tensor(new_state).float().unsqueeze(0)
		self.memory.push((self.last_state, new_state, torch.LongTensor([int(self.last_action)]), torch.Tensor([self.last_reward])))
		action = self.select_action(new_state)
		if len(self.memory.memory) > 100:
		    batch_state, batch_next_state, batch_action, batch_reward = self.memory.sample(100)
		    self.learn(np.array(batch_state), np.array(batch_next_state), batch_reward, batch_action)
		self.last_action = action
		self.last_state = new_state
		self.last_reward = reward
		self.reward_window.append(reward)
		if len(self.reward_window) > 1000:
		    del self.reward_window[0]
		return action
	def score(self):
		return sum(self.reward_window)/(len(self.reward_window)+1.)