#ifndef __BLOB_HPP__
#define __BLOB_HPP__
#include <vector>
#include <armadillo>
using std::vector;
using arma::cube;
using std::string;

/*
Blob�������������������ǽ����ݼ�ȫ�����뵽�ڴ浱�С�
Blob��   n   ��  c  ��h��w ��
     (������Ŀ��ͨ�������ߣ���)
Blob��������һ����ΪBlob��cube���͵�vector����洢
*/

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
	Blob() : n(0), c(0), w(0), h(0){}
	Blob(const int n, const int c, const int w, const int h, int type = TDEFAULT);  //���غ���
	vector<cube> &GetData();
	cube & operator[](int index);
	void Print(string str = "");
	Blob SubBlob(int low_index, int high_index);
	void Init(const int n, const int c, const int w, const int h, int type);
	int GetC();
	int GetH();
	int GetW();
private:
	int n;
	int c;
	int h;
	int w;
	vector<cube> blob_data;
	
};


#endif // __BLOB_HPP__
