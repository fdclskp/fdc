#include<bits/stdc++.h>
#define INF 2000000000
#define MOD 1000000007
#define EPS (1e-10)

using namespace std;


vector<int>  dx = {1,-1,0,0};
vector<int>  dy = {0,0,1,-1};


int main() {
	

	fstream ifs("data.gsp");
	string str;

	//vector<string> graph_label;
	vector<int> graph_label;
	vector<vector<int>> neighbor_array;
	while(getline(ifs,str)){
		if (str[0] == 't') {
			neighbor_array.clear();
			neighbor_array.shrink_to_fit();
			//graph_label.push_back(str.substr(7,str.size()-1));
			graph_label.push_back(str[6]-'0');
		}else if (str[0] == 'v') {
			//node_label.push_back(str.substr(4,4));
			vector<int> v = {(str[2]-'0')};
			neighbor_array.push_back(v);
		}else if (str[0] == 'e') {
			neighbor_array[(str[2]-'0')].push_back((str[4]-'0'));
			neighbor_array[(str[4]-'0')].push_back((str[2]-'0'));
		}
		if (str == "") {
			for (int i = 0; i < neighbor_array.size(); i++) {
				for (int j = 0; j < neighbor_array[i].size(); j++) {
					cout << neighbor_array[i][j] << " ";
				}
				cout << endl;
			}
			cout << endl;
		}
	}

	return 0;

}
