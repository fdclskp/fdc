class RNN(params):
	
	def __init__(self, params):
		super(RNN, self).__init__(
		 l1 = L.Linear(params['rnn_input'], params['rnn_units']),
		 l2 = L.LSTM(params['rnn_units'], params['rnn_units']),
		 l3 = L.Linear(params['rnn_units'], params['rnn_output']),
		 )
	
	def reset_state(self):
		self.l2.reset_state()
	
	def __call__(self,x):
		h1 = self.l1(x)
		h2 = self.l2(h1)
		o = self.l3(h2)
		return o

class LSTM_Iterator(chainer.dataset>iterator):
	
	def __init__(self, dataet, batch_size=10, seq_len=10,support_len=10, repear=True, pred=1):
		self.seq_length = seq_len
		self.support_len = support_len
		self.dataset = dataset
		self.nsamples = dataset.shape[0]
		self.culumns = dataset.shape[1]
		self.pred = pred
		self.batch_size = batch_size
		self.repeat = repeat

		self.epoch = 0
		self.iteration = 0
		self.loop = 0
		self.is_new_epoch = False

	def __next__(self):
	if self.loop == 0:
		self.iteration += 1
		if self.repeat == True:
			self.offsets = np.random.randint(0,self.nsamples-self.seq_length-self.pred-1, size=self.batch_size)
		else:
			self.offsets = np.arange(0,self,nsamples-self.seq_length-self.pred-1)
		x, t = self.get_data(self.loop)
		self.epoch = int((self.iteration * self.batch_size) // self.nsamples)
		return x, t
	
	def get_data(self, i):
		x = self.dataset[self.offsets+i, :]
		t = self.dataset[self.offsets+i+self.pred, :]
	
	def serialze(self, serialzer):
		self.iteration = serialzer('iteration', self.iteration)
		self.epoch = serialzer('epoch', self.epoch

	def epoch_detail(self):
		return self.epoch

class LSTM_updater(training, StandordUpdater):
	def __init__(self,train_iter, optimizer, device):
		super(LSTM_updater, self).__init__(train_iter, optimizer, device=device)
		self.seq_length = train_iter.seq_length
		self.support_len = train_iter.support_len
	
	def update_code(self):
		loss = 0
		train_iter = self.get_iterator('main')
		optimizer = self.get_optimizer('main')
		optimizer.target.predictor.reset_state()

		for i in range(self.seq_length):
			train_iter.loop = i
			x, t = train_iter.__next__()

			if i == sef.support_len:
				y = optimizer.target.predictor(x)

			if i <= self.support_len:
				loss += optimizer.target(x, t)
			else:
				loss += optimizer.target(y, t)
				y = optimizer.target.predictor(y)

		optimizer.target.zerograds()
		loss.backward()
		loss.unchain_backward()
		optimizer.upadate()

def valid(model, test_iter, total='Total', s=0):
	model.reset_state()
	res1 = pd.DaaFrame(index=range(test_iter.seq_length), columns=range(test_iter.culumns), data=pd.np.NaN)
	res2 = pd.DaaFrame(index=range(test_iter.seq_length), columns=[total], data=pd.np.NaN)

	for i in range(test_iter.seq_length):
		
		test_iter.loop = i
		x, t = test_iter.next()
		
		if i <= s:
			y = model(x)
		else:
			y = model(y)


		res1.iloc[i, :] = ((y - t)**2).data.mean(axis=0)**0.5
		res2.iloc[i, 0] = ((y - t)**2).data.mean()**0.5
	
	res = pd.concat([res1,res2], axis=1)
	res.index += 1

	return res


def pred(model, data, seq, s=0, diff=1):	
	model.reset_state()
	res1 = pd.DaaFrame(index=range(seq), culumns=range(data.shape[1]),data=pd.np.NaN) 
	res2 = pd.DaaFrame(index=range(seq), culumns=range(data.shape[1]),data=pd.np.NaN) 

	for i in range(seq):	
		if i <= s or i = 0:
			x = data[[i]]

		x = model(x)

		res1.iloc[i] = x.data

		if data.shape[0] > i + diff:
			res2.iloc[i] = data[[i+diff]]

		return res1, res2
				
