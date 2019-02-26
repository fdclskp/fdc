import os

def load_data(path):
	for filename in os.listdir(path):
		f = open(filename)	
	return 
		
def convert_from_gsp(filename):
	
	label_data = []
	input_data = []
	g = Graph()
	for line in open(filename, 'r'):
		line = line[:-1]
		line = line.split(' ')
		if line[0] == 't':
			label_data.append(int(line[3]))
		elif line[0] == 'v':
			g.adj_list.append([int(line[2])])
		elif line[0] == 'e':
			g.adj_list[int(line[1])].append(g.adj_list[int(line[2])][0])
			g.adj_list[int(line[2])].append(g.adj_list[int(line[1])][0])
		elif line[0] == "":
			input_data.append(g)	
			g = Graph()
	return np.array(data)
