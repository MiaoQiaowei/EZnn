#include "Blob.hpp"
#include "cassert"
using namespace std;
using namespace arma;


Blob::Blob(const int n, const int c, const int w, const int h, int type) : n(n), c(c), w(w), h(h)
{
	arma_rng::set_seed_random();  //ϵͳ�����������(���û����һ�䣬�ͻ�ÿ����������(����)ʱ��Ĭ�ϴ�����1��ʼ�������������
	Init(n, c, w, h, type);
}

Blob::Blob(const vector<int> shape, int type) : n(shape[0]), c(shape[1]), w(shape[2]), h(shape[3])
{
	arma_rng::set_seed_random();  //ϵͳ�����������(���û����һ�䣬�ͻ�ÿ����������(����)ʱ��Ĭ�ϴ�����1��ʼ�������������
	Init(n, c, w, h, type);
}

void Blob::Init(const int n, const int c, const int w, const int h, int type)
{

	if (type == TONES)
	{
		blob_data = vector<cube>(n, cube(h, w, c, fill::ones));
		return;
	}
	if (type == TZEROS)
	{
		blob_data = vector<cube>(n, cube(h, w, c, fill::zeros));
		return;
	}
	if (type == TDEFAULT)
	{
		blob_data = vector<cube>(n, cube(h, w, c));
		return;
	}
	if (type == TRANDU)
	{
		for (int i = 0; i < n; ++i)   //����n����������ֵ�����ȷֲ�����cube���ѵ���vector����
			blob_data.push_back(arma::randu<cube>(h, w, c)); //�ѵ�
		return;
	}
	if (type == TRANDN)
	{
		for (int i = 0; i < n; ++i)   //����n����������ֵ(��׼��˹�ֲ�����cube���ѵ���vector����
			blob_data.push_back(arma::randn<cube>(h, w, c)); //�ѵ�
		return;
	}

}

void Blob::Print(string str)
{
	assert(!blob_data.empty());  //���ԣ�   blob_data��Ϊ�գ�������ֹ����
	cout << str << endl;
	for (int i = 0; i < n; ++i)  //N_Ϊblob_data��cube����
	{
		printf("N = %d\n", i);
		this->blob_data[i].print();//��һ��ӡcube������cube�����غõ�print()
	}
}

vector<cube> &Blob::GetData()
{
	return Blob::blob_data;
}

cube & Blob::operator[](int index)
{
	return Blob::blob_data[index];
}

Blob & Blob::operator=(const double in)
{
	for (int i = 0; i < n; i++)
	{
		blob_data[i].fill(in);
	}
	return *this;
}

Blob Blob::SubBlob(int low_index, int high_index)
{
	
	if (low_index <= high_index)
	{
		Blob temp(high_index - low_index, c, h, w);
		for (int i = low_index; i < high_index; i++)
		{
			temp[i-low_index] = (*this)[i];
		}
		return temp;
	}
	else
	{
		Blob temp(n + high_index - low_index, c, h, w);
		for (int i = low_index; i < n; i++)
		{
			temp[i - low_index] = (*this)[i];
		}
		for (int i = 0; i < high_index; i++)
		{
			temp[i+n-low_index] = (*this)[i];
		}
		return temp;
	}
}

Blob & Blob::operator*=(const double in)
{
	for (int i = 0; i < n; i++)
	{
		blob_data[i] = blob_data[i] * in;
	}
	return *this;
}

Blob Blob::Pad(int pad, double val)
{
	assert(!blob_data.empty());
	Blob x_paded(n, c, w + pad * 2, h + pad * 2);
	x_paded = val; 
	for (int n_ = 0; n_< n; n_++)
	{
		for (int c_ = 0; c_ < c; c_++)
		{
			for (int h_ = 0; h_ < h; h_++)
			{
				for (int w_ = 0; w_ < w; w_++)
				{
					x_paded[n_]( h_ + pad,  w_ + pad, c_) = blob_data[n_]( h_, w_, c_);
				}
			}
		}

	}
	return x_paded;
}

void Blob::Max(double in)
{
	assert(!blob_data.empty());
	for (int i = 0; i < n; i++)
	{
		blob_data[i].transform([in](double e) {return e > in ? e : in; });
	}
}

Blob Blob::DeletePad(int pad)
{
	assert(!blob_data.empty());   //���ԣ�Blob����Ϊ��
	//int N_ = n;
	//int C_ = c;
	//int  W_ = w;
	//int H_ = h;
	Blob out(n, c, w - 2 * pad, h - 2 * pad);
	for (int n_ = 0; n_ < n; ++n_)
	{
		for (int c_ = 0; c_ < c; ++c_)
		{
			for (int h_ = pad; h_ < h - pad; ++h_)
			{
				for (int w_ = pad; w_ < w - pad; ++w_)
				{
					//ע�⣬out�������Ǵ�0��ʼ�ģ�����Ҫ��ȥpad
					out[n_](h_ - pad, w_ - pad, c_) = blob_data[n_](h_, w_, c_);
				}
			}
		}
	}
	return out;
}

vector<int> Blob::size() const
{
	vector<int> shape{ n,c,w,h };
	return shape;
}

Blob operator*(Blob& A, Blob& B)  //��Ԫ�����ľ���ʵ�֣�����û�����޶����� (Blob& Blob::)������ʽ
{
	//(1). ȷ����������Blob�ߴ�һ��
	vector<int> size_A = A.size();
	vector<int> size_B = B.size();
	for (int i = 0; i < 4; ++i)
	{
		assert(size_A[i] == size_B[i]);  //���ԣ���������Blob�ĳߴ磨N,C,H,W��һ����
	}
	//(2). �������е�cube��ÿһ��cube����Ӧλ����ˣ�cube % cube��
	int N = size_A[0];
	Blob C(A.size());
	for (int i = 0; i < N; ++i)
	{
		C[i] = A[i] % B[i];
	}
	return C;
}



int Blob::GetC() 
{
	return c;
}

int Blob::GetH() 
{
	return h;
}

int Blob::GetW() 
{
	return w;
}

int Blob::GetN()
{
	return n;
}