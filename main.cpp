#include "myNet.hpp"
#include <iostream>
#include <string>

using namespace std;



int main(int argc, char** argv)
{
	string configFile = "./myModel.json";
	NetParam net_param;
	//1.��ȡmyModel.json���ڴ���
	net_param.readNetParam(configFile);

	//2.��ӡ���ǵ���Щ����
	cout << "learning rate =  " << net_param.lr << endl;
	cout << "batch size =  " << net_param.batch_size << endl;

	vector<string> layers_ = net_param.layers;
	vector<string> ltypes_ = net_param.ltypes;
	for (int i = 0; i < layers_.size(); ++i)
	{
		cout << "layer = " << layers_[i] << " ; " << "ltype = " << ltypes_[i] << endl;
	}

	system("pause");

}