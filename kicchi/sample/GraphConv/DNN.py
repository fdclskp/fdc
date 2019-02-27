import chainer
from chainer import Chain, Link, training
from chainer.training import extensions
import chainer.links as L
import chainer.functions as F
import numpy as np


def build_weights(self, params, in_num):
	initializer = chainer.initializers.HeNormal()
	for i in range(in_num):
		setattr(self,'output_weights_' + str(i), L.Linear(params['num_features'],1))

class DNN(Chain):
	def __init__(self, params, in_num):
		super(DNN, self).__init__()
		with self.init_scope():
			build_weights(self, params, in_num)	
	
	def __call__(self,x,y):
		return F.mean_squared_error(self.forward(x),y)
	
	def get_weights(self,i):
		return "self.output_weights_" + str(i)

	def forward(self, x, adj_list):
		feartures = np.empty((1,len(adj_list[0])), dtype=np.float32)
		for i in range(len(x)):
			fearture = np.empty((1,1), dtype=np.float32)
			output_weights = eval(self.get_weights(i))
			for j in range(len(adj_list[i])):
				output_weights = eval(self.get_weights(i))
				f = output_weights(x[i][j])
				fearture = F.concat((fearture, f),axis=1)
			cut = F.reshape(fearture[0][1:],(1,len(fearture[0][1:])))
			feartures = F.concat((feartures, cut), axis=0)
		cuts = feartures[1:]
		return cuts
	


