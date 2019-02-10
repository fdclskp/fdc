#README
#予測モデルの呼び出し
#訓練
#グラフの期間の指定
#ファイルの指定
#とか
import numpy as np
import glob
import pandas as pd
import chainer
from chainer import cuda, Function, Variable, optimizers
from chainer import Link, Chain
import chainer.functions as F
import chainer.links as L
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from GraphConv import load_data
from GraphConv import SGC
from GraphConv import RNN
from GraphConv import lstm_func as lf



#使用するファイル
year = '2018'
DATA_DIR = "indicator/stock_2016-2018/stock_indicator" + year + "/daily/"

#ハイパーパラメータ
model_params = {'rnn_input'  : 2,
				'rnn_units'  : 10,
				'rnn_output' : 1,
				'num_features' : 6,
				'gnn_layers' : 1,
				}
train_params = {'epochs' : 3,
				}
#TODO
#load_data

stock_data = dict()
class SGCRN(Chain):
	def __init__(self, model_params):	
		super(SGCRN, self).__init__(
			sgc = SGC.SGC(model_params),
			rnn = RNN.RNN(model_params),	
			#f = None
		)


	def __call__(self, x, y):
		pred = self.prediction(x)
		return F.mean_squared_error(pred,y)

	def prediction(self,x):
		#self.f = self.sgc(x, stock_data)
		#pred = self.rnn(self.f)
		print(x)
		pred = self.rnn(x)
		return pred		
			

def train_model(Model, train_params, x_train, y_train, x_val=None, y_val=None):
	training_curve = []
	optimizre = oprimizer.Adam()
	oprimizer.setup(Model)
	
	for epoch in range(train_params['epochs']):
		Model.zerograds()		
		loss = Model(x_train,y_train)
		loss.backward()
		optimizre.update()
		print("TRAINING OK!")
		import pdb;pdb.set_trace()
		
		cur_loss = loss.data[0]
		training_curve.append(cur_loss)
		print("Iteration ",epoch, ": train loss ", cur_loss)
		#validation も後で追加
	
	return Model, training_curve

def getSineData():
    
    N_data = 200
    N_Loop = 4
    t = np.linspace(0, 2*np.pi*N_Loop, num=N_data)
    
    X = 0.8*np.sin(2.0*t)
    Y = 0.8*np.cos(1.0*t)
    
    N_train = int(N_data*0.75)
#    N_test = int(N_data*0.25)
    
    DataSet = np.c_[X, Y].astype(np.float32)
    
    train, test = np.array(DataSet[:N_train]), np.array(DataSet[N_train:])
    return train, test


#def convert_from_gsp(filename):
#	
#	label_data = []
#	neighbor_data = []
#	
#	data = np.array([])
#	g = Graph()
#	for line in open(filename, 'r'):
#		line = line[:-1]
#		line = line.split(' ')
#		if line[0] == 't':
#			label_data.append(line[3])
#		elif line[0] == 'v':
#			data = np.append(data, line[1])
#			g.nodes = np.append(g.nodes, int(line[1]))
#			#print('v')				
#		elif line[0] == 'e':
#			g.nodes = g.nodes.T
#			print(g.nodes)
#			g.nodes[int(line[2])] = np.append(g.nodes[int(line[2])], int(line[1]))
#			#g.nodes[] = np.append(g.nodes, line[1])
#			#data[int(line[1])].append(int(line[2]))
#			#data[int(line[2])].append(int(line[1]))
#		elif line[0] == "":
#			data = np.append(data, g)	
#			g = Graph()
#
#	for i in range(len(data)):
#		print(data[i])
#
#	return data

class Graph():
	nodes = np.array([[]])
	delta = np.array([])

def array_convert(filename):
	#後で一つにまとめる(array_y)
	graph_array = np.array([])
	g = Graph()
	node_list = []
	for line in open(filename, 'r'):
		if line == '\n':
			#print("in """)
			graph_array = np.append(graph_array, g)
			g.nodes = np.array(node_list)
			#print(g.nodes)
			g = Graph()
			node_list = []
		else:
			line = line[:-1].split(' ')
			line = [int(s) for s in line]
			node_list.append(line)
		
	return graph_array

def array_y(filename):
	y_array = []
	y = []
	for line in open(filename, 'r'):
		if line == '\n':
			#print(y)
			y_array.append(y)
			y = []
		else:
			line = line[:-1]
			y.append(line)

	return np.array(y_array)

#def concat_data(X, Y):
#
#	for i in  range(len(X)):
#		for j in range(len(Y[i])):	
#			X[i].nodes[j] = np.insert(X[i].nodes[j], 0, Y[i][j])
#
#	return X


def main():
	#print(year,"'s data")
	#traindata,valdata,testdata = load_data(DATA_DIR)
	#x_train, y_train = traindata
	#x_val, y_val = valdata
	#x_test, y_test = testdata
	train, test = getSineData()
	#dataset = convert_from_gsp('data.gsp')	
	#x_dataset = array_convert('data_list.txt')
	#y_dataset = array_y('data_y.txt')
	##dataset = concat_data(x_dataset, y_dataset)
	#x_train = x_dataset[:int(len(x_dataset)*0.7)]
	#x_train = x_dataset[:2]
	#x_test = x_dataset[int(len(x_dataset)*0.7):]
	#y_train = y_dataset[:int(len(y_dataset)*0.7)]
	#y_train = y_dataset[:2]
	#y_test = y_dataset[int(len(y_dataset)*0.7):]
	#(x_train, x_test, y_train, y_test) = train_test_split(
    #x_dataset, y_dataset, test_size=0.3,)
	for line in open('data_features.txt', "r"):
		line = line[:-1].split(',')
		line = [int(s) for s in line]
		stock_data[line[0]] = line[1:]

	def run_experiment():
		Model = SGCRN(model_params)	

		trained_Model, training_curve = \
			train_model(Model,
						x_train, y_train
						)

		evaluation = trained_Model(x_test,y_test) 
		return evaluation, training_curve

	Model = SGCRN(model_params)
	#output = Model(x_train, y_train)
	output = Model.prediction(train)

	out = 'result_1'


#   グラフから特徴ベクトル抽出できたら以下を実行。
#	#learn
#	trainer = lf.getTrainer(train, n_units=100, gpu=-1, batch_size=10, seq_len=10, support_len=10, pred=1,
#	                        out=out, snap=10, epoch=50)
#	
#	lf.chainer.config.train = True
#	
#	models = glob.glob(out+"/trainer*")
#	models.sort()
#	
#	if len(models) == 0:
#		trainer.run()
#	else:
#		lf.serializers.load_npz(models[-1], trainer)
#		trainer.updater.get_iterator('main').iteration = trainer.updater.iteration
#		trainer.run()
#	
#	#pred
#	#trainer = lf.getTrainer(train, n_units, gpu=-1, batch_size=10, seq_len=10, support_len=10, pred=1,
#	#                        out=out, snap=10, epoch=100)
#	
#	test_iter = lf.LSTM_Iterator(test, repeat=False, seq_len=20)
#	
#	trainers = glob.glob(out+"/trainer*")
#	trainers.sort()
#	
#	lf.serializers.load_npz(trainers[-1], trainer)
#	trainer.updater.get_iterator('main').iteration = trainer.updater.iteration
#	
#	model = trainer.updater.get_optimizer('main').target.predictor
#	
#	predres = lf.pred(model, np.array([[0, 1.0]]).astype(np.float32), 50, 0, 1)
#	
#	pd.concat([predres[0], predres[1]], axis=1).plot(legend=False)
#	plt.show()
#
	#print("Starting experiment...")
	#test_loss, training_curve = run_experiment()
	#print
	#print("test evaluation", test_loss)
	
	
if __name__ == '__main__':
	main()
