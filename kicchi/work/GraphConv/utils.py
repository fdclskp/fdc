from .SGC import Graph
def concat_data(X, Y):

	for i in  range(len(X)):
		for j in range(len(Y[i])):	
			X[i].adj_list[j] = np.insert(X[i].nodes[j], 0, Y[i][j])

	return X

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

def array_convert(filename):
	#後で一つにまとめる(array_y)
	graph_array = []
	node_list = []
	g = Graph()
	for line in open(filename, 'r'):
		if line == '\n':
			g.adj_list = node_list
			graph_array.append(g)
			g = Graph()
			node_list = []
		elif line[4] == '-':
			continue
		else:
			line = line[:-1].split(' ')
			line = [int(s) for s in line]
			node_list.append(line)
		
	return graph_array

def array_y(filename):
	y_data = []
	y = []
	for line in open(filename, 'r'):
		if line[0] == '\n':
			y_data.append(y)
			y = []
		else:
			line = line[:-1]
			
			y.append(float(line))

	return y_data
