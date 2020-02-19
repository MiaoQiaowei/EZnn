#include "Net.hpp"
#include "Blob.hpp"
#include <iostream>
#include <string>
#include <memory>
#include "Mnist.h"

using namespace std;

int main(int argc, char** argv)
{
	string ModelPath = "./Model.json";
	string imagesPath = "D:\\github\\EZnn\\Minist\\train\\train-images.idx3-ubyte";
	string labelsPath = "D:\\github\\EZnn\\Minist\\train\\train-labels.idx1-ubyte";
	Mnist mnist(imagesPath, labelsPath, ModelPath);
	mnist.JsonTest();//测试模型状态
	mnist.MnistTest();//读取数据
	//输出三组数据进行展示
	vector<cube>&imagesList = mnist.images->get_data();
	vector<cube>&labesList = mnist.images->get_data();
	for (int i = 0; i < 3; i++)
	{
		imagesList[i].print("images:");
		labesList[i].print("labels");
	}

}