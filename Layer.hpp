#ifndef  __LAYER_HPP__
#define __LAYER_HPP__
#include <iostream>
#include <memory>
#include "Blob.hpp"

using std::vector;
using std::shared_ptr;


struct LayerParam  //结构体 （层参数，主要是每一层的细节部分）
{
	/*1.卷积层超参数 */
	int conv_stride;
	int conv_pad;
	int conv_width;
	int conv_height;
	int conv_kernels;

	/*2.池化层超参数*/
	int pool_stride;
	int pool_width;
	int pool_height;

	/*3.全连接层超参数（即该层神经元个数） */
	int fc_kernels;
};


class Layer
{
public:
	Layer() {};
	~Layer() {};
	virtual	void Init(const vector<int>&input_shape, vector<shared_ptr<Blob>>&data, const LayerParam param, const string name) {};
	virtual	void CalculateShape(const vector<int>&input_shape, vector<int>&out_shape, LayerParam param)=0;
	virtual void forward(const vector<shared_ptr<Blob>>&in, shared_ptr<Blob>&out, const LayerParam &param)=0;

private:

};

class Conv:public Layer
{
public:
	Conv() {};
	~Conv() {};
	void Init(const vector<int>&input_shape, vector<shared_ptr<Blob>>&data, const LayerParam param, const string name);
	void CalculateShape(const vector<int>&input_shape, vector<int>&out_shape, LayerParam param);
	void forward(const vector<shared_ptr<Blob>>&in, shared_ptr<Blob>&out, const LayerParam &param);
private:

};

class Relu :public Layer
{
public:
	Relu() {};
	~Relu() {};
	void Init(const vector<int>&input_shape, vector<shared_ptr<Blob>>&data, const LayerParam param, const string name);
	void CalculateShape(const vector<int>&input_shape, vector<int>&out_shape, LayerParam param);
	void forward(const vector<shared_ptr<Blob>>&in, shared_ptr<Blob>&out, const LayerParam &param);
private:

};


class Fc :public Layer
{
public:
	Fc() {};
	~Fc() {};
	void Init(const vector<int>&input_shape, vector<shared_ptr<Blob>>&data, const LayerParam param, const string name);
	void CalculateShape(const vector<int>&input_shape, vector<int>&out_shape, LayerParam param);
	void forward(const vector<shared_ptr<Blob>>&in, shared_ptr<Blob>&out, const LayerParam &param);
private:

};

class Pool :public Layer
{
public:
	Pool() {};
	~Pool() {};
	void Init(const vector<int>&input_shape, vector<shared_ptr<Blob>>&data, const LayerParam param, const string name);
	void CalculateShape(const vector<int>&input_shape, vector<int>&out_shape, LayerParam param);
	void forward(const vector<shared_ptr<Blob>>&in, shared_ptr<Blob>&out, const LayerParam &param);
private:

};

class Softmax
{
public:
	static void softmax_cross_entropy_with_logits(const vector<shared_ptr<Blob>>& in, double& loss, shared_ptr<Blob>& diff_out);
};

#endif  //__LAYER_HPP__