#include "myBlob.hpp"
#include "cassert"
using namespace std;
using namespace arma;


Blob::Blob(const int n, const int c, const int h, const int w, int type) : N_(n), C_(c), H_(h), W_(w)
{
	arma_rng::set_seed_random();  //ϵͳ�����������(���û����һ�䣬�ͻ�ÿ����������(����)ʱ��Ĭ�ϴ�����1��ʼ�������������
	_init(N_, C_, H_, W_, type);

}

void Blob::_init(const int n, const int c, const int h, const int w, int type)
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

void Blob::print(string str)
{
	assert(!blob_data.empty());  //���ԣ�   blob_data��Ϊ�գ�������ֹ����
	cout << str << endl;
	for (int i = 0; i < N_; ++i)  //N_Ϊblob_data��cube����
	{
		printf("N = %d\n", i);
		this->blob_data[i].print();//��һ��ӡcube������cube�����غõ�print()
	}
}