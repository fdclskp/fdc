#README
import numpy as np
import glob, os
import pandas as pd
import chainer
from chainer import cuda, Function, Variable, optimizers,serializers
from chainer import Link, Chain
import chainer.functions as F
import chainer.links as L
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from GraphConv import load_data, array_y, array_convert
from GraphConv import SGC, Graph
from GraphConv import RNN
from GraphConv import DNN
from GraphConv import lstm_func as lf



#使用するファイル
year = '2018'
DATA_DIR = "indicator/stock_2016-2018/stock_indicator" + year + "/daily/"

#ハイパーパラメータ
model_params = dict(rnn_input = 2,
				rnn_units = 10,
				rnn_output = 1,
				num_features = 17,
				gnn_layers = 3,
				)
train_params = dict(epochs = 101,
				)

stock_datas = []
class SGCRN(Chain):
	def __init__(self, model_params):	
		super(SGCRN, self).__init__(
			sgc = SGC.SGC(model_params),
			dnn = DNN.DNN(model_params,5000),	
		)
	def __call__(self, x, y, params):
		pred, _ = self.prediction(x,params)
		y = Variable(np.array(y, dtype=np.float32).reshape(len(y),1,225))
		return F.mean_squared_error(pred,y)

	def prediction(self,x,params):
		self.f, adj_list = self.sgc(x, stock_datas, params)
		pred = self.dnn.forward(self.f,adj_list)
		return pred	,adj_list	
			

def train_model(Model,model_params, train_params, x_train, y_train, x_val=None, y_val=None):
	training_curve = []
	optimizer = optimizers.Adam()
	optimizer.setup(Model)
	optimizer.add_hook(chainer.optimizer.WeightDecay(0.0001))
	for epoch in range(train_params['epochs']):
		Model.zerograds()		
		loss = Model(x_train,y_train,model_params)
		loss.backward()
		optimizer.update()
		cur_loss = loss._data[0]
		training_curve.append(cur_loss)
		if (epoch % 10  == 0):
			serializers.save_npz("result_1/epoch_" + str(epoch) + "_fdcmodel.npz", Model) 
		print("Iteration ",epoch, ": train loss ", cur_loss)
	
	return Model, training_curve

def main():
	x_dataset = array_convert('data_list.txt')
	y_dataset = array_y('data_y.txt')

	x_train = x_dataset[:(len(x_dataset)-1)]
	x_test  = x_dataset[len(x_dataset)-1:]
	y_train = y_dataset[:len(y_dataset)-1]
	y_test  = y_dataset[len(y_dataset)-1:]


	
	for	filename in glob.glob('./features/*'):
		stock_data = dict()
		for line in open(filename, "r"):
			line = line[:-1].split(',')
			for i in range(len(line)):
				if line[i] != '':
					line[i] = float(line[i])
				else:
					line[i] = 0
			stock_data[str(int(line[0]))] = line[1:]
		stock_datas.append(stock_data)

	def run_experiment():
		Model = SGCRN(model_params)	
		trained_Model, training_curve = train_model(Model,model_params, train_params, x_train, y_train)
		evaluation = trained_Model(x_test,y_test, model_params) 
		return trained_Model, evaluation, training_curve


	trained_Model, evaluation, training_curve = run_experiment()
	pred, adj_list = trained_Model.prediction(x_test,model_params)
	for i in range(len(pred[0][0])):
		print("CODE : ", adj_list[0][i][0], " -> ", pred[0][0][i]._data[0])



if __name__ == '__main__':
	main()
