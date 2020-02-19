#ifndef __BLOB_HPP__
#define __BLOB_HPP__
#include <vector>
#include <armadillo>
using std::vector;
using arma::cube;
using std::string;

enum FillType
{

	TONES = 1,   //cube所有元素都填充为1
	TZEROS = 2,  //cube所有元素都填充为0
	TRANDU = 3,  //将元素设置为[0,1]区间内均匀分布的随机值
	TRANDN = 4,  //使用μ= 0和σ= 1的标准高斯分布设置元素
	TDEFAULT = 5

};


//Blob a;
//Blob a(10,3,3,3,TONES);
class Blob
{
public: //构造函数：这个类通过调用构造函数来实例化对象
	Blob() : N_(0), C_(0), H_(0), W_(0)
	{}
	Blob(const int n, const int c, const int h, const int w, int type = TDEFAULT);  //重载函数
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
