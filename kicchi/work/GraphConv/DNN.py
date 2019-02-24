import chainer
from chainer import Chain, Link, training, Variable
from chainer.training import extensions
import chainer.links as L
import chainer.functions as F
import numpy as np


def build_weights(self, params, in_num):
	initializer = chainer.initializers.HeNormal()
	#for i in range(in_num):
	#	setattr(self,'output_weights_' + str(i), L.Linear(params['num_features'],1))
	setattr(self,'output_weights', L.Linear(params['num_features'],1, initialW=initializer))

class DNN(Chain):
	def __init__(self, params, in_num):
		super(DNN, self).__init__()
		with self.init_scope():
			build_weights(self, params, in_num)	
	
	def __call__(self,x,y):
		print(F.mean_squared_error(self.forward(x),y))
		return F.mean_squared_error(self.forward(x),y)
	
	def get_weights(self):
		return "self.output_weights"

	def forward(self, x, adj_list):
		output_weights = eval(self.get_weights())
		ret = Variable(np.empty((1,1,225), dtype=np.float32))
		for i in range(len(x)):
			output = Variable(np.empty((1,1), dtype=np.float32))
			for j in range(len(x[i])):
				y = output_weights(F.reshape(x[i][j],(1,len(x[i][j]))))
				output = F.concat((output, y),axis=0)
			ret = F.concat((ret, F.reshape(output[1:],(1,1,225))), axis=0)
		return ret[1:]	
