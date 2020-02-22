#include"Layer.hpp"
#include <cassert>
#include<opencv2/opencv.hpp>

using namespace std;
using namespace arma;



void Conv::Init(const vector<int>&input_shape, vector<shared_ptr<Blob>>&data, const LayerParam param, const string name)
{
	//获取尺寸
	int kernel_num = param.conv_kernels;
	int c = input_shape[1];
	int w = param.conv_width;
	int h = param.conv_height;

	//进行初始化
	if (!data[1])//W初始化
	{
		data[1].reset(new Blob(kernel_num, c, w, h, TRANDN));
		(*data[1]) *= 1e-2;
	}
	if (!data[2])//B初始化
	{
		data[2].reset(new Blob(kernel_num, 1, 1, 1, TRANDN));
		(*data[2]) *= 1e-2;
	}

	cout << "Conv Init" << endl;
	return;
}

void Conv::CalculateShape(const vector<int>&input_shape, vector<int>&out_shape, LayerParam param)
{
	//获取输入Blob尺寸
	int n_in = input_shape[0];
	int c_in = input_shape[1];
	int w_in = input_shape[2];
	int h_in = input_shape[3];

	//获取卷积核尺寸
	int n_ = param.conv_kernels;
	int h_ = param.conv_height;
	int w_ = param.conv_width;
	int p_ = param.conv_pad;
	int s_ = param.conv_stride;

	//输出Blob尺寸
	int n_out = n_in;
	int c_out = n_;
	int h_out = (h_in + 2 * p_ - h_) / s_ + 1;
	int w_out = (w_in + 2 * p_ - w_) / s_ + 1;

	out_shape[0] = n_out;
	out_shape[1] = c_out;
	out_shape[2] = w_out;
	out_shape[3] = h_out;
	return;
}

void Conv::forward(const vector<shared_ptr<Blob>>&in, shared_ptr<Blob>&out, const LayerParam &param)
{
	cout << "Conv forward" << endl;
	
	if (out)
		out.reset();
		
	//获取相关尺寸
	assert(in[0]->GetC() == in[1]->GetC());
	int n_x = in[0]->GetN();
	int c_x = in[0]->GetC();
	int w_x = in[0]->GetW();
	int h_x = in[0]->GetH();

	int n_w = in[1]->GetN();
	int w_w = in[1]->GetW();
	int h_w = in[1]->GetH();

	int w_out = (w_x + 2 * param.conv_pad - w_w) / param.conv_stride + 1;
	int h_out = (h_x + 2 * param.conv_pad - h_w) / param.conv_stride + 1;
	//pading操作
	Blob x_paded = in[0]->Pad(param.conv_pad);

	//卷积操作
	out.reset(new Blob(n_x, n_w, w_out, h_out));
	for (int n_ = 0; n_ < n_x; n_++)
	{
		for (int c_ = 0; c_ < n_w; c_++)
		{
			for (int h_ = 0; h_ < h_out; h_++)
			{
				for (int w_ = 0; w_ < w_out; w_++)
				{
					cube window = x_paded[n_](span(h_*param.conv_stride, h_*param.conv_stride + h_w - 1), 
											  span(w_*param.conv_stride, w_*param.conv_stride + w_w - 1), 
											  span::all);
					//out = Wx+b
					(*out)[n_](h_, w_, c_) = arma::accu(window % (*in[1])[c_]) + as_scalar((*in[2])[c_]);
				}
			}
		}
	}

	
	(*in[1])[0].slice(0).print("W=");
	cout << "b:" << endl;
	cout << as_scalar((*in[2])[0]) << endl;
	(*out)[0].slice(0).print("Out=");
	
	//开始卷积运算
}

void Fc::Init(const vector<int>&input_shape, vector<shared_ptr<Blob>>&data, const LayerParam param, const string name)
{
	//获取尺寸
	int kernel_num = param.fc_kernels;
	int c = input_shape[1];
	int w = input_shape[2];
	int h = input_shape[3];

	//进行初始化
	if (!data[1])//W初始化
	{
		data[1].reset(new Blob(kernel_num, c, w, h, TRANDN));
		(*data[1]) *= 1e-2;
	}
	if (!data[2])//B初始化
	{
		data[2].reset(new Blob(kernel_num, 1, 1, 1, TRANDN));
		(*data[2]) *= 1e-2;
	}

	cout << "Fc Init" << endl;
	return;
}

void Fc::CalculateShape(const vector<int>&input_shape, vector<int>&out_shape, LayerParam param)
{
	//获取卷积核尺寸
	int n_out = input_shape[0];
	int c_out = param.fc_kernels;
	int h_out = 1;
	int w_out = 1;

	out_shape[0] = n_out;
	out_shape[1] = c_out;
	out_shape[2] = w_out;
	out_shape[3] = h_out;
	return;
}

void Fc::forward(const vector<shared_ptr<Blob>>&in, shared_ptr<Blob>&out, const LayerParam &param)
{
	cout << "Fc forward" << endl;
}

void Pool::Init(const vector<int>&input_shape, vector<shared_ptr<Blob>>&data, const LayerParam param, const string name)
{
	cout << "Pool Init" << endl;
	return;
}

void Pool::CalculateShape(const vector<int>&input_shape, vector<int>&out_shape, LayerParam param)
{
	//获取输入Blob尺寸
	int n_in = input_shape[0];
	int c_in = input_shape[1];
	int w_in = input_shape[2];
	int h_in = input_shape[3];

	//获取卷积核尺寸
	int h_ = param.pool_height;
	int w_ = param.pool_width;
	int s_ = param.pool_stride;

	//输出Blob尺寸
	int n_out = n_in;
	int c_out = c_in;
	int h_out = (h_in - h_) / s_ + 1;
	int w_out = (w_in - w_) / s_ + 1;

	out_shape[0] = n_out;
	out_shape[1] = c_out;
	out_shape[2] = w_out;
	out_shape[3] = h_out;
	return;
}

void Pool::forward(const vector<shared_ptr<Blob>>&in, shared_ptr<Blob>&out, const LayerParam &param)
{
	cout << "Pool forward" << endl;
}

void Relu::Init(const vector<int>&input_shape, vector<shared_ptr<Blob>>&data, const LayerParam param, const string name)
{
	cout << "Relu Init" << endl;
	return;
}

void Relu::CalculateShape(const vector<int>&input_shape, vector<int>&out_shape, LayerParam param)
{
	out_shape.assign(input_shape.begin(), input_shape.end());
	return;
}

void Relu::forward(const vector<shared_ptr<Blob>>&in, shared_ptr<Blob>&out, const LayerParam &param)
{
	cout << "Relu forward" << endl;
}