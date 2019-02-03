#README
#予測モデルの呼び出し
#訓練
#グラフの期間の指定
#ファイルの指定
#とか
import numpy as np
import chainer
from chainer import cuda, Function, Variable, optimizers
from chainer import Link, Chain
import chainer.functions as F
import chainer.links as L

from GraphConv import load_data
from GraphConv import SGC
from GraphConv import RNN


#使用するファイル
year = '2018'
DATA_DIR = "indicator/stock_2016-2018/stock_indicator" + year + "/daily/"

#ハイパーパラメータ
model_params = {}
train_params = {'epochs' : 10,
				}
#TODO
#load_data

class SGCRN(Chain):
	def __init__(self, model_params):	
		super(SGCRN, self).__init__(
			sgc = SGC(model_params),
			rnn = RNN(model_params),	
		)

	def __call__(self, x, y):
		pred = self.prediction(x)
		return F.mean_squared_error(pred,y)
			
	def prediction(self,x):
		f = self.SGC(x)
		pred = self.rnn(f)
		return pred		

def train_model(Model, train_params, x_train, y_train, x_val, y_val):
	training_curve = []
	optimizre = oprimizer.Adam()
	oprimizer.setup(Model)
	
	for epoch in range(train_params['epochs']):
		Model.zerograds()		
		loss = Model(x_train,y_train)
		loss.backward()
		optimizre.update()
		
		cur_loss = loss.data[0]
		training_curve.append(cur_loss)
		print("Iteration ",epoch, ": train loss ", cur_loss)
		#validation も後で追加
	
	return Model, training_curve
	
def main():
	print(year,"'s data")
	traindata,valdata,testdata = load_data(DATA_DIR)
	x_train, y_train = traindata
	x_val, y_val = valdata
	x_test, y_test = testdata
	def run_experiment():
		Model = SGCRN(model_params)	

		trained_Model, training_curve = \
			train_model(Molde,
						x_train, y_train,
						x_val, y_val)

		evaluation = trained_Model(x_test,y_test) 
		return evaluation, training_curve


	print("Starting experiment...")
	test_loss, training_curve = run_experiment()
	print
	print("test evaluation", test_loss)
	
	
if __name__ == '__main__':
	main()
