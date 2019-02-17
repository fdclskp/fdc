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
				num_features = 6,
				gnn_layers = 1,
				)
train_params = dict(epochs = 3,
				)

stock_datas = []
class SGCRN(Chain):
	def __init__(self, model_params):	
		super(SGCRN, self).__init__(
			sgc = SGC.SGC(model_params),
			dnn = DNN.DNN(model_params,5000),	
		)
	def __call__(self, x, y, params):
		pred = self.prediction(x,params)
		y = Variable(np.array(y, dtype=np.float32))
		return F.mean_squared_error(pred,y)

	def prediction(self,x,params):
		self.f, adj_list = self.sgc(x, stock_datas, params)
		pred = self.dnn.forward(self.f,adj_list)
		return pred		
			

def train_model(Model,model_params, train_params, x_train, y_train, x_val=None, y_val=None):
	training_curve = []
	optimizer = optimizers.Adam()
	optimizer.setup(Model)
	for epoch in range(train_params['epochs']):
		Model.zerograds()		
		loss = Model(x_train,y_train,model_params)
		loss.backward()
		optimizer.update()
		cur_loss = loss._data[0]
		training_curve.append(cur_loss)
		print("Iteration ",epoch, ": train loss ", cur_loss)
		#validation も後で追加
	
	return Model, training_curve

def main():
	x_dataset = array_convert('data_list.txt')
	y_dataset = array_y('data_y.txt')
	x_train = np.array(x_dataset[:int(len(x_dataset)*0.7)])
	x_test = np.array(x_dataset[int(len(x_dataset)*0.7):])
	y_train = np.array(y_dataset[:int(len(y_dataset)*0.7)])
	y_test = np.array(y_dataset[int(len(y_dataset)*0.7):])

	
	for	filename in os.dirlist('features') 
		stock_data = dict()
		for line in open(filename, "r"):
			line = line[:-1].split(',')
			line = [int(s) for s in line]
			stock_data[str(line[0])] = line[1:]
		stock_datas.append(stock_data)

	def run_experiment():
		Model = SGCRN(model_params)	
		trained_Model, training_curve = train_model(Model,model_params, train_params, x_train, y_train)
		evaluation = trained_Model(x_test,y_test, model_params) 
		return evaluation, training_curve

	out = 'result_1'

	evaluation, training_curve = run_experiment()


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
