#ifndef  __Mnist_HPP__
#define __Mnist_HPP__

#include "Net.hpp"
#include "Blob.hpp"
#include <iostream>
#include <string>
#include <memory>


using std::string;

class Mnist
{
public:
	string imagesPath;
	string labelsPath;
	string JsonPath;
	NetParam net;
	vector<string> layers_;
	vector<string> ltypes_;
	Blob* images;
	Blob* labels;

	Mnist(string imagesPath,string labelsPath, string JsonPath);
	~Mnist();
	void MnistTest();
	void JsonTest();

private:
	int  ReverseInt(int i);
	void ReadMnistData(string path, Blob* &images);
	void ReadMnistLabel(string path, Blob* &labels);
};

#endif // __Minist_HPP__