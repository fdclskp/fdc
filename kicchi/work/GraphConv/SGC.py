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
			
			numpy_code_features = np.empty((len(code_features),len(code_features[0]),len(code_features[0][0]),1), dtype=np.float32)
			for i in range(len(code_features)):
				for j in range(len(code_features[i])):
					for k in range(len(code_features[i][j])):
						numpy_code_features[i][j][k] = code_features[i][j][k]
			return numpy_code_features, adj_list
		
		def update_layer(self, code_features, adj_list):#未完
			hidden_weight = eval('self.self_activation')
			print((code_features))
			self_activation_features = hidden_weight(code_features)
			#print((code_features))
			#self_activation_features = hidden_weight(code_features)
			adj_weight = eval('self.adj_weight')
			ret = []
			for i in range(len(adj_list)): #nodeのID
				f0 = F.concat((self_activation_features[i], self_activation_features[i]), axis=0)
				f1 = F.expand_dims(f0,axis=0)
				f2 = F.relu(adj_weight(f1))
				pooling_feature = f2
				for j in range(len(adj_list[i])-1):# i 番目のノードの隣接する数
					for k in range(len(adj_list)):
						if (adj_list[k][0] == adj_list[j+1]):
							f0 = F.concat((self_activation_features[i], self_activation_features[adj_list[j+1]]), axis=0)
							f1 = F.expand_dims(f0,axis=0)
							pooling_feature = pooling_feature + F.relu(adj_weight(f1))
				pooling_feature = pooling_feature / len(adj_list[i])
				#ret.append(pooling_feature)
			return ret[0]


		
		def daily_features_func(self, graphs):
			code_features, adj_lists= read_features(self, graphs)
			for layer in range(params['gnn_layers']):
				#ret = []
				#for i in range(len(code_features)):
				code_features = update_layer(self, code_features, adj_lists)
					#ret.append(code_features[i])
				#code_features = ret
			return code_features, adj_lists
		daily_features, adj_lists = daily_features_func(self,x)
		return daily_features, adj_lists
