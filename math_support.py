import math
import numpy as np
def cost_fun(y,a):
	return -1*(y*math.log(a)+(1-y)*math.log(1-a))
def loss(x,y):
		return(np.sum(y-x))
def sigmoidprime(s):
	return s*(1-s)
def sigmoid(x):
	return 1.0 / (1.0 + np.exp(-1.0 * x))