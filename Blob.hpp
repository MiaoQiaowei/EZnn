#ifndef __BLOB_HPP__
#define __BLOB_HPP__
#include <vector>
#include <armadillo>
using std::vector;
using arma::cube;
using std::string;

enum FillType
{

	TONES = 1,   //cube����Ԫ�ض����Ϊ1
	TZEROS = 2,  //cube����Ԫ�ض����Ϊ0
	TRANDU = 3,  //��Ԫ������Ϊ[0,1]�����ھ��ȷֲ������ֵ
	TRANDN = 4,  //ʹ�æ�= 0�ͦ�= 1�ı�׼��˹�ֲ�����Ԫ��
	TDEFAULT = 5

};


//Blob a;
//Blob a(10,3,3,3,TONES);
class Blob
{
public: //���캯���������ͨ�����ù��캯����ʵ��������
	Blob() : N_(0), C_(0), H_(0), W_(0)
	{}
	Blob(const int n, const int c, const int h, const int w, int type = TDEFAULT);  //���غ���
	vector<cube> &get_data();
	cube & operator[](int index);
	void print(string str = "");
private:
	void _init(const int n, const int c, const int h, const int w, int type);
private:
	int N_;
	int C_;
	int H_;
	int W_;
	vector<cube> blob_data;
	
};


#endif // __BLOB_HPP__
