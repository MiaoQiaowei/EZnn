#ifndef __BLOB_HPP__
#define __BLOB_HPP__
#include <vector>
#include <armadillo>
using std::vector;
using arma::cube;
using std::string;

/*
Blob是数据流，他的作用是将数据集全部读入到内存当中。
Blob（   n   ，  c  ，w,  h ）
     (样本数目，通道数，宽，高)
Blob整体是由一个名为Blob的cube类型的vector数组存储
*/

enum FillType
{
	TONES = 1,   //cube所有元素都填充为1
	TZEROS = 2,  //cube所有元素都填充为0
	TRANDU = 3,  //将元素设置为[0,1]区间内均匀分布的随机值
	TRANDN = 4,  //使用μ= 0和σ= 1的标准高斯分布设置元素
	TDEFAULT = 5
};


class Blob
{
public: //构造函数：这个类通过调用构造函数来实例化对象
	Blob() : n(0), c(0), w(0), h(0){}
	Blob(const int n, const int c, const int w, const int h, int type = TDEFAULT) ;  //重载函数
	Blob(const vector<int> shape_, int type = TDEFAULT);
	Blob & operator*=(const double in);
	Blob & operator=(const double in);
	Blob SubBlob(int low_index, int high_index);
	Blob Pad(int pad, double val = 0);
	Blob DeletePad(int pad);
	cube & operator[](int index);
	friend Blob operator*(Blob& A, Blob& B);
	friend Blob operator*(double lr, Blob& B);
	friend Blob operator+(Blob& a_, Blob& b_);
	int GetC();
	int GetH();
	int GetW();
	int GetN();
	vector<cube> &GetData();
	vector<int> size() const;
	void Print(string str);
	void Max(double in = 0.0);
	void Init(const int n, const int c, const int w, const int h, int type);

private:
	int n;
	int c;
	int h;
	int w;
	vector<cube> blob_data;
};


#endif // __BLOB_HPP__
