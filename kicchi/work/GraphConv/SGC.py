import numpy as np

import chainer 
from chainer import cuda, Function, Variable, optimizers, initializers, Link, Chain
import chainer.functions as F
import chainer.links as L

class Graph():
	adj_list = []
	delta = []

def build_weights(self, params):
	initializer = chainer.initializers.HeNormal()
	#setattr(self, 'params', params,)

	'''hidden_weight '''
	num_indicator = params['num_features']
	setattr(self, 'self_activation', L.Linear(num_indicator, num_indicator, initialW=initializer))
					
	'''adj_weight'''
	setattr(self, 'adj_weight', L.Linear(2 * num_indicator, num_indicator, initialW=initializer))


class SGC(Chain):
	def __init__(self, params):
		super(SGC,self).__init__()
		with self.init_scope():
			build_weights(self,params)
				
	def __call__(self, x, stock_datas, params):

		def read_features(self, graphs):
			adj_list = []
			code_features = []
			for i in range(len(graphs)):
				adj_list.append(graphs[i].adj_list)
			for i in range(len(graphs)):
				code_feature = []
				for j in range(len(adj_list[i])):
					code_feature.append(stock_datas[i][str(adj_list[i][j][0])])
				code_features.append(code_feature)
			numpy_code_features = np.array(code_features, dtype=np.float32)
			return numpy_code_features, adj_list
		
		def update_layer(self, code_features, adj_list):#未完
			#self_activation_features = np.empty(code_features.shape)
			#self_activation_features = []
			#print((code_features.shape))
			#import pdb;pdb.set_trace()
			#for i in range(len(code_features[0])):
				#h = hidden_weight(np.array(code_features[0][i]).reshape(1,17))
				#print(type(code_features[0][i]))
				#print("before hidden")
				#h = hidden_weight(((code_features[0][i])))
				#print(h.shape)
				#self_activation_features.append(h)
				#self_activation_features[i] = h
			#self_activation_features = hidden_weight(code_features)
			#print((code_features))
			#self_activation_features = hidden_weight(code_features)
			'''hidden_weight一回やめる'''
			#hidden_weight = eval('self.self_activation')
			#self_activation_features = []
			#for i in range(len(code_features)):
			#	self_activation_features.append(hidden_weight(code_features[i]))
			#print(code_features.shape)
			#print(self_activation_features[0].shape)
			def func(adj_list,i,j):
				target = adj_list[j][0]
				f = Variable(np.empty((1,17),dtype=np.float32))
				for k in range(len(adj_list[j])):
					for l in range(len(adj_list)):
						if adj_list[j][k] == adj_list[l][0]:
							f0 = F.concat((code_features[i][j], code_features[i][l]), axis=0)
							f1 = adj_weight(F.reshape(f0,(1,len(f0))))
							f2 = F.relu(f1)
							f += f2 
				sum_f = F.sum(f, axis=0) / len(adj_list[j])
				#print(sum_f)
				return sum_f


			adj_weight = eval('self.adj_weight')
			ret = Variable(np.empty((1,225,17),dtype=np.float32))
			for i in range(len(adj_list)): # for G in all graph 
				g = Variable(np.empty((1,17),dtype=np.float32))
				for j in range(len(adj_list[i])):# for v in G
					pooling_feature = F.reshape(func(adj_list[i],i,j), (1,17))
					g = F.concat((g, pooling_feature),axis=0)
				g = F.reshape(g[1:], (1,225,17))
				ret = F.concat((ret, g),axis=0)
			return ret[1:]
		
		def daily_features_func(self, graphs):
			code_features, adj_lists= read_features(self, graphs)
			code_features = Variable(code_features)
			for layer in range(params['gnn_layers']):
				code_features  = update_layer(self, code_features, adj_lists)
			return code_features, adj_lists
		daily_features, adj_lists = daily_features_func(self,x)
		return daily_features, adj_lists
