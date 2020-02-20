#ifndef  __LAYER_HPP__
#define __LAYER_HPP__
#include <iostream>
#include <vector>
#include <memory>
#include "Blob.hpp"

using std::vector;
using std::shared_ptr;


struct LayerParam  //�ṹ�� �����������Ҫ��ÿһ���ϸ�ڲ��֣�
{
	/*1.����㳬���� */
	int conv_stride;
	int conv_pad;
	int conv_width;
	int conv_height;
	int conv_kernels;

	/*2.�ػ��㳬����*/
	int pool_stride;
	int pool_width;
	int pool_height;

	/*3.ȫ���Ӳ㳬���������ò���Ԫ������ */
	int fc_kernels;
};


class Layer
{
public:
	Layer() {};
	~Layer() {};
	virtual	void Init(const vector<int>&input_shape, const vector<shared_ptr<Blob>>&data, LayerParam param, const string name) {};

private:

};

class Conv:public Layer
{
public:
	Conv() {};
	~Conv() {};
	void Init(const vector<int>&input_shape, const vector<shared_ptr<Blob>>&data, LayerParam param, const string name);
private:

};

class Relu :public Layer
{
public:
	Relu() {};
	~Relu() {};
	void Init(const vector<int>&input_shape, const vector<shared_ptr<Blob>>&data, LayerParam param, const string name);
private:

};


class Fc :public Layer
{
public:
	Fc() {};
	~Fc() {};
	void Init(const vector<int>&input_shape, const vector<shared_ptr<Blob>>&data, LayerParam param, const string name);
private:

};

class Pool :public Layer
{
public:
	Pool() {};
	~Pool() {};
	void Init(const vector<int>&input_shape, const vector<shared_ptr<Blob>>&data, LayerParam param, const string name);
private:

};

#endif  //__LAYER_HPP__