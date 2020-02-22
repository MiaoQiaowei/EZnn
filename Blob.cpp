#include "Blob.hpp"
#include "cassert"
using namespace std;
using namespace arma;


Blob::Blob(const int n, const int c, const int w, const int h, int type) : n(n), c(c), w(w), h(h)
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