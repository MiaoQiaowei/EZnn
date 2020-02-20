#include"Layer.hpp"
#include<memory>
#include<iostream>
#include<vector>
#include"Blob.hpp"
#include<unordered_map>
using namespace std;
using std::unordered_map;
using std::vector;

void Conv::Init(const vector<int>&input_shape, const vector<shared_ptr<Blob>>&data, LayerParam param, const string name)
{
	cout << "Conv Init" << endl;
}

void Fc::Init(const vector<int>&input_shape, const vector<shared_ptr<Blob>>&data, LayerParam param, const string name)
{
	cout << "Fc Init" << endl;
}

void Pool::Init(const vector<int>&input_shape, const vector<shared_ptr<Blob>>&data, LayerParam param, const string name)
{
	cout << "Pool Init" << endl;
}

void Relu::Init(const vector<int>&input_shape, const vector<shared_ptr<Blob>>&data, LayerParam param, const string name)
{
	cout << "Relu Init" << endl;
}