#include "Blob.hpp"
#include "cassert"
using namespace std;
using namespace arma;


Blob::Blob(const int n, const int c, const int h, const int w, int type) : n(n), c(c), h(w), w(h)
{
	arma_rng::set_seed_random();  //ϵͳ�����������(���û����һ�䣬�ͻ�ÿ����������(����)ʱ��Ĭ�ϴ�����1��ʼ�������������
	Init(n, c, h, w, type);

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