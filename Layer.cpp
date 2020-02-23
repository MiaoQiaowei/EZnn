#include"Layer.hpp"
#include <cassert>
#include<opencv2/opencv.hpp>

using namespace std;
using namespace arma;

void Conv::Init(const vector<int>&input_shape, vector<shared_ptr<Blob>>&data, const LayerParam param, const string name)
{
	//��ȡ�ߴ�
	int kernel_num = param.conv_kernels;
	int c = input_shape[1];
	int w = param.conv_width;
	int h = param.conv_height;

	//���г�ʼ��
	if (!data[1])//W��ʼ��
	{
		data[1].reset(new Blob(kernel_num, c, w, h, TRANDN));
		(*data[1]) *= 1e-2;
	}
	if (!data[2])//B��ʼ��
	{
		data[2].reset(new Blob(kernel_num, 1, 1, 1, TRANDN));
		(*data[2]) *= 1e-2;
	}

	cout << "Conv Init" << endl;
	return;
}

void Conv::CalculateShape(const vector<int>&input_shape, vector<int>&out_shape, LayerParam param)
{
	//��ȡ����Blob�ߴ�
	int n_in = input_shape[0];
	int c_in = input_shape[1];
	int w_in = input_shape[2];
	int h_in = input_shape[3];

	//��ȡ����˳ߴ�
	int n_ = param.conv_kernels;
	int h_ = param.conv_height;
	int w_ = param.conv_width;
	int p_ = param.conv_pad;
	int s_ = param.conv_stride;

	//���Blob�ߴ�
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
		
	//��ȡ��سߴ�
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
	//pading����
	Blob x_paded = in[0]->Pad(param.conv_pad);

	//�������
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
	
	//��ʼ�������
}

void Conv::backward(const shared_ptr<Blob>& diff_in, const vector<shared_ptr<Blob>>& cache, vector<shared_ptr<Blob>>& gradient, const LayerParam& param)
{
	cout << "Conv backward" << endl;
	
	//step1. ��������ݶ�Blob�ĳߴ磨dX---grads[0]��
	gradient[0].reset(new Blob(cache[0]->size(), TZEROS));
	gradient[1].reset(new Blob(cache[1]->size(), TZEROS));
	gradient[2].reset(new Blob(cache[2]->size(), TZEROS));
	//step2. ��ȡ�����ݶ�Blob�ĳߴ磨diff_in��
	int n_x = diff_in->GetN();         //�����ݶ�Blob��cube��������batch����������
	int c_x = diff_in->GetC();         //�����ݶ�Blobͨ����
	int h_x = diff_in->GetH();         //�����ݶ�Blob��
	int w_x = diff_in->GetW();         //�����ݶ�Blob��
	
	//step3. ��ȡ�������ز���
	int h_w = param.conv_height;
	int w_w = param.conv_width;
	int stride_w = param.conv_stride;

	//step4. ������
	Blob pad_x = cache[0]->Pad(param.conv_pad);             //����ʵ�ʷ��򴫲������Ӧ��������������Blob
	Blob gradient_x(pad_x.size(), TZEROS);                      //�ݶ�BlobӦ����ò������Blob�ߴ籣��һ��

	//step5. ��ʼ���򴫲�
	for (int n_ = 0; n_ < n_x; ++n_)   //���������ݶ�din��������
	{
		for (int c_ = 0; c_ < c_x; ++c_)  //���������ݶ�din��ͨ����
		{
			for (int h_ = 0; h_ < h_x; ++h_)   //���������ݶ�din�ĸ�
			{
				for (int w_ = 0; w_ < w_x; ++w_)   //���������ݶ�din�Ŀ�
				{
					//(1). ͨ���������ڣ���ȡ��ͬ��������Ƭ��
					cube window = pad_x[n_](span(h_*stride_w, h_*stride_w + h_w - 1), span(w_*stride_w, w_*stride_w + w_w - 1), span::all);
					//(2). �����ݶ�
					//dX
					gradient_x[n_](span(h_*stride_w, h_*stride_w + h_w - 1), span(w_*stride_w, w_*stride_w + w_w - 1), span::all) += (*diff_in)[n_](h_, w_, c_) * (*cache[1])[c_];
					//dW  --->grads[1]
					(*gradient[1])[c_] += (*diff_in)[n_](h_, w_, c_) * window / n_x;
					//db   --->grads[2]
					(*gradient[2])[c_](0, 0, 0) += (*diff_in)[n_](h_, w_, c_) / n_x;
				}
			}
		}
	}
	////���Դ���
	//(*diff_in)[0].slice(0).print("input:   diff_in=");				    //�����ݶȣ���ӡ��һ��din�ĵ�һ������
	//(*diff_in)[0].slice(1).print("input:   diff_in=");				    //�����ݶȣ���ӡ��һ��din�ĵڶ�������
	//(*diff_in)[0].slice(2).print("input:   diff_in=");				    //�����ݶȣ���ӡ��һ��din�ĵ���������
	//(*cache[1])[0].slice(0).print("W1=");		                //��ӡ��һ������˵ĵ�һ������	
	//(*cache[1])[1].slice(0).print("W2=");		                //��ӡ�ڶ�������˵ĵ�һ������	
	//(*cache[1])[2].slice(0).print("W3=");		                //��ӡ����������˵ĵ�һ������		
	//gradient_x[0].slice(0).print("output:   gradient_x=");		//����ݶȣ���ӡ��һ��pad_dX�ĵ�һ������

	//step6. ȥ������ݶ��е�padding����
	(*gradient[0]) = gradient_x.DeletePad(param.conv_pad);
}




void Fc::Init(const vector<int>&input_shape, vector<shared_ptr<Blob>>&data, const LayerParam param, const string name)
{
	//��ȡ�ߴ�
	int kernel_num = param.fc_kernels;
	int c = input_shape[1];
	int w = input_shape[2];
	int h = input_shape[3];

	//���г�ʼ��
	if (!data[1])//W��ʼ��
	{
		data[1].reset(new Blob(kernel_num, c, w, h, TRANDN));
		(*data[1]) *= 1e-2;
	}
	if (!data[2])//B��ʼ��
	{
		data[2].reset(new Blob(kernel_num, 1, 1, 1, TRANDN));
		(*data[2]) *= 1e-2;
	}

	cout << "Fc Init" << endl;
	return;
}

void Fc::CalculateShape(const vector<int>&input_shape, vector<int>&out_shape, LayerParam param)
{
	//��ȡ����˳ߴ�
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
	if (out)
		out.reset();

	//��ȡ��سߴ�
	assert(in[0]->GetC() == in[1]->GetC());
	int n_x = in[0]->GetN();
	int c_x = in[0]->GetC();
	int w_x = in[0]->GetW();
	int h_x = in[0]->GetH();

	int n_w = in[1]->GetN();
	int w_w = in[1]->GetW();
	int h_w = in[1]->GetH();
	assert(h_x == h_w && w_x == w_w);

	int w_out = 1;
	int h_out = 1;

	//�������
	out.reset(new Blob(n_x, n_w, w_out, h_out));
	for (int n_ = 0; n_ < n_x; n_++)
	{
		for (int c_ = 0; c_ < n_w; c_++)
		{
			(*out)[n_](0,0, c_) = arma::accu((*in[0])[n_] % (*in[1])[c_]) + as_scalar((*in[2])[c_]);
		}
	}

	//��ʼ�������
}

void Fc::backward(const shared_ptr<Blob>& diff_in, const vector<shared_ptr<Blob>>& cache, vector<shared_ptr<Blob>>& gradient, const LayerParam& param)
{
	cout << "Fc backward" << endl;
	gradient[0].reset(new Blob(cache[0]->size(), TZEROS));
	gradient[1].reset(new Blob(cache[1]->size(), TZEROS));
	gradient[2].reset(new Blob(cache[2]->size(), TZEROS));

	int n_x = gradient[0]->GetN();
	int f_x = gradient[1]->GetN();
	assert(f_x == cache[1]->GetN());

	for (int n_ = 0; n_ < n_x; ++n_)
	{
		for (int f_ = 0; f_ < f_x; ++f_)
		{
			//dX
			(*gradient[0])[n_] += (*diff_in)[n_](0, 0, f_) * (*cache[1])[f_];
			//dW
			(*gradient[1])[f_] += (*diff_in)[n_](0, 0, f_) * (*cache[0])[n_] / n_x;
			//db
			(*gradient[2])[f_] += (*diff_in)[n_](0, 0, f_) / n_x;
		}
	}
}




void Pool::Init(const vector<int>&input_shape, vector<shared_ptr<Blob>>&data, const LayerParam param, const string name)
{
	cout << "Pool Init" << endl;
	return;
}

void Pool::CalculateShape(const vector<int>&input_shape, vector<int>&out_shape, LayerParam param)
{
	//��ȡ����Blob�ߴ�
	int n_in = input_shape[0];
	int c_in = input_shape[1];
	int w_in = input_shape[2];
	int h_in = input_shape[3];

	//��ȡ����˳ߴ�
	int h_ = param.pool_height;
	int w_ = param.pool_width;
	int s_ = param.pool_stride;

	//���Blob�ߴ�
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
	if (out)
		out.reset();

	//��ȡ��سߴ�
	int n_x = in[0]->GetN();
	int c_x = in[0]->GetC();
	int w_x = in[0]->GetW();
	int h_x = in[0]->GetH();
	
	int h_w = param.pool_height;
	int w_w = param.pool_width;

	int w_out = (w_x - w_w) / param.pool_stride + 1;
	int h_out = (h_x - h_w) / param.pool_stride + 1;

	//�������
	out.reset(new Blob(n_x, c_x, w_out, h_out));
	for (int n_ = 0; n_ < n_x; n_++)
	{
		for (int c_ = 0; c_ < c_x; c_++)
		{
			for (int h_ = 0; h_ < h_out; h_++)
			{
				for (int w_ = 0; w_ < w_out; w_++)
				{
					(*out)[n_](h_, w_, c_) = (*in[0])[n_]
					(
						span(h_*param.pool_stride, h_*param.pool_stride + h_w - 1),
						span(w_*param.pool_stride, w_*param.pool_stride + w_w - 1),
						span(c_,c_)
					).max();
				}
			}
		}
	}
}

void Pool::backward(const shared_ptr<Blob>& diff_in, const vector<shared_ptr<Blob>>& cache, vector<shared_ptr<Blob>>& gradient, const LayerParam& param)
{
	cout << "Pool  backward " << endl;
	//step1. ��������ݶ�Blob�ĳߴ磨dX---grads[0]��
	gradient[0].reset(new Blob(cache[0]->size(), TZEROS));
	//step2. ��ȡ�����ݶ�Blob�ĳߴ磨din��
	int n_x = diff_in->GetN();        //�����ݶ�Blob��cube��������batch����������
	int c_x = diff_in->GetC();         //�����ݶ�Blobͨ����
	int h_x = diff_in->GetH();      //�����ݶ�Blob��
	int w_x = diff_in->GetW();    //�����ݶ�Blob��

	//step3. ��ȡ�ػ�����ز���
	int h_p = param.pool_height;
	int w_p = param.pool_width;
	int stride_p = param.pool_stride;

	//step4. ��ʼ���򴫲�
	for (int n_ = 0; n_ < n_x; ++n_)   //���cube��
	{
		for (int c_ = 0; c_ < c_x; ++c_)  //���ͨ����
		{
			for (int h_ = 0; h_ < h_x; ++h_)   //���Blob�ĸ�
			{
				for (int w_ = 0; w_ < w_x; ++w_)   //���Blob�Ŀ�
				{
					//(1). ��ȡ����mask
					mat window = (*cache[0])[n_](span(h_*param.pool_stride, h_*param.pool_stride + h_p - 1),
												 span(w_*param.pool_stride, w_*param.pool_stride + w_p - 1),
												 span(c_, c_));
					double maxv = window.max();
					mat mask = conv_to<mat>::from(maxv == window);  //"=="���ص���һ��umat���͵ľ���umatת��Ϊmat
					//(2). �����ݶ�
					(*gradient[0])[n_](span(h_*param.pool_stride, h_*param.pool_stride + h_p - 1),
									   span(w_*param.pool_stride, w_*param.pool_stride + w_p - 1),
									   span(c_, c_)) += mask * (*diff_in)[n_](h_, w_, c_);  //umat  -/-> mat
				}
			}
			//(*diff_in)[0].slice(0).print("diff_in=");
			//(*cache[0])[0].slice(0).print("cache=");  //mask
			//(*gradient[0])[0].slice(0).print("gradient=");
		}
	}
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
	if (out)
		out.reset();
	out.reset( new Blob(*in[0]));
	out->Max(0);
	cout << "Relu forward" << endl;
}

void Relu::backward(const shared_ptr<Blob>& diff_in, const vector<shared_ptr<Blob>>& cache, vector<shared_ptr<Blob>>& gradient, const LayerParam& param)
{
	cout << "Relu backward" << endl;
	//step1. ��������ݶ�Blob�ĳߴ磨dX---grads[0]��
	gradient[0].reset(new Blob(*cache[0]));

	//step2. ��ȡ����mask
	int n_x = gradient[0]->GetN();
	for (int n_ = 0; n_ < n_x; ++n_)
	{
		(*gradient[0])[n_].transform([](double e) {return e > 0 ? 1 : 0; });
	}
	(*gradient[0]) = (*gradient[0]) * (*diff_in);

	//(*diff_in)[0].slice(0).print("diff_in=");				//�����ݶȣ���ӡ��һ��din�ĵ�һ������
	//(*cache[0])[0].slice(0).print("cache=");		//���룺 ��ӡ��һ��cache�ĵ�һ������
	//(*gradient[0])[0].slice(0).print("gradient=");		//����ݶȣ���ӡ��һ��grads�ĵ�һ������
	return;
}






void Softmax::softmax_cross_entropy_with_logits(const vector<shared_ptr<Blob>>& in, double& loss, shared_ptr<Blob>& diff_out)
{
	cout << "Softmax..." << endl;
	if (diff_out)
		diff_out.reset();

	//��ȡ��سߴ�
	
	int n_x = in[0]->GetN();
	int c_x = in[0]->GetC();
	int w_x = in[0]->GetW();
	int h_x = in[0]->GetH();

	assert(w_x == 1 && h_x == 1);
	loss =0.0;
	diff_out.reset(new Blob(n_x, c_x, w_x, h_x));
	for (int i = 0; i < n_x; i++)
	{
		cube prob = arma::exp((*in[0])[i])/arma::accu(arma::exp((*in[0])[i]));
		loss += (-arma::accu((*in[1])[i] % arma::log(prob)));
		(*diff_out)[i] = prob - (*in[1])[i];
	}
	loss /= n_x;

 }
