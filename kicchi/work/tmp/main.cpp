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

	while(getline(ifs,str)){
		if (str[0] == 'v') {
			cout << rand() % 10 << " ";
		}else if (str == "") {
			cout << endl;
		}
	}
	return 0;
}
