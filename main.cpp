#include "Net.hpp"
#include "Blob.hpp"
#include <iostream>
#include <string>
#include <memory>
#include "Mnist.h"

using namespace std;
/*

//minist 读取操作
int ReverseInt(int i)  ////把大端数据转换为我们常用的小端数据 （大小端模式的原因）
{
	unsigned char ch1, ch2, ch3, ch4;  //一个int有4个char
	ch1 = i & 255;
	ch2 = (i >> 8) & 255;
	ch3 = (i >> 16) & 255;
	ch4 = (i >> 24) & 255;
	return ((int)ch1 << 24) + ((int)ch2 << 16) + ((int)ch3 << 8) + ch4;
}

void ReadMnistData(string path, shared_ptr<Blob> &images)
{
	ifstream file(path, ios::binary);
	if (file.is_open())
	{
		//mnist原始数据文件中32位的整型值是大端存储，C/C++变量是小端存储，所以读取数据的时候，需要对其进行大小端转换!!!!
		//1.从文件中获知魔术数字（一般都是起到标识的作用，比如用来判断这个文件是不是MNIST里面的train-labels.idx1-ubyte文件），图片数量和图片宽高信息
		int magic_number = 0;
		int number_of_images = 0;
		int n_rows = 0;
		int n_cols = 0;
		file.read((char*)&magic_number, sizeof(magic_number));
		magic_number = ReverseInt(magic_number);  //高低字节调换
		cout << "magic_number=" << magic_number << endl;
		file.read((char*)&number_of_images, sizeof(number_of_images));
		number_of_images = ReverseInt(number_of_images);
		cout << "number_of_images=" << number_of_images << endl;
		file.read((char*)&n_rows, sizeof(n_rows));
		n_rows = ReverseInt(n_rows);
		cout << "n_rows=" << n_rows << endl;
		file.read((char*)&n_cols, sizeof(n_cols));
		n_cols = ReverseInt(n_cols);
		cout << "n_cols=" << n_cols << endl;

		//2.将图片转为Blob存储！
		for (int i = 0; i<number_of_images; ++i)  //遍历所有图片
		{
			for (int h = 0; h<n_rows; ++h)   //遍历高
			{
				for (int w = 0; w<n_cols; ++w)   //遍历宽
				{
					unsigned char temp = 0;
					file.read((char*)&temp, sizeof(temp));      //读入一个像素值！		
					//-----将temp中的数据写入Blob中------
					(*images)[i](h, w, 0) = (double)temp / 255;
					//-----------------------------------

				}
			}
		}
	}
	else
	{
		cout << "no data file found :-(" << endl;
	}

}
void ReadMnistLabel(string path, shared_ptr<Blob> &labels)
{

	ifstream file(path, ios::binary);
	if (file.is_open())
	{
		//1.从文件中获知魔术数字，图片数量
		int magic_number = 0;
		int number_of_images = 0;
		file.read((char*)&magic_number, sizeof(magic_number));
		magic_number = ReverseInt(magic_number);
		cout << "magic_number=" << magic_number << endl;
		file.read((char*)&number_of_images, sizeof(number_of_images));
		number_of_images = ReverseInt(number_of_images);
		cout << "number_of_Labels=" << number_of_images << endl;
		//2.将所有标签转为Blob存储！（手写数字识别：0~9）
		for (int i = 0; i<number_of_images; ++i)
		{
			unsigned char temp = 0;
			file.read((char*)&temp, sizeof(temp));
			//-----将temp中的数据写入Blob中------
			(*labels)[i](0, 0, (int)temp) = 1;

			//-----------------------------------
		}
	}
	else
	{
		cout << "no label file found :-(" << endl;
	}
}
*/

int main(int argc, char** argv)
{
	/*
	string configFile = "./Model.json";

	NetParam net_param;
	//1.读取myModel.json到内存中
	net_param.readNetParam(configFile);

	//2.打印参数,检查JSON文件的读取情况
	cout << "learning rate =  " << net_param.lr << endl;
	cout << "batch size =  " << net_param.batch_size << endl;

	vector<string> layers_ = net_param.layers;
	vector<string> ltypes_ = net_param.ltypes;
	for (int i = 0; i < layers_.size(); ++i)
	{
		cout << "layer = " << layers_[i] << " ; " << "ltype = " << ltypes_[i] << endl;
	}

	//实例化一个blob对象
	Blob test_blob(2, 3, 5, 5, TONES);
	test_blob.print();

	//实例化两个Blob对象
	shared_ptr<Blob> images (new Blob(60000, 1, 28, 28, TONES));
	shared_ptr<Blob> labels(new Blob(60000, 10, 1, 1, TONES));
	//数据路径
	ReadMnistData("D:\\github\\EZnn\\Minist\\train\\train-images.idx3-ubyte", images);
	ReadMnistLabel("D:\\github\\EZnn\\Minist\\train\\train-labels.idx1-ubyte", labels);

	vector<cube>&imagesList = images->get_data();
	vector<cube>&labesList = labels->get_data();

	for (int i = 0; i < 3; i++)
	{
		imagesList[i].print("images:");
		labesList[i].print("labels");
	}

	system("pause");
	*/
	string ModelPath = "./Model.json";
	string imagesPath = "D:\\github\\EZnn\\Minist\\train\\train-images.idx3-ubyte";
	string labelsPath = "D:\\github\\EZnn\\Minist\\train\\train-labels.idx1-ubyte";
	Mnist mnist(imagesPath, labelsPath, ModelPath);
	mnist.JsonTest();//
	mnist.MnistTest();
	vector<cube>&imagesList = mnist.images->get_data();
	vector<cube>&labesList = mnist.images->get_data();

	for (int i = 0; i < 3; i++)
	{
		imagesList[i].print("images:");
		labesList[i].print("labels");
	}

}