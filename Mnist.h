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
	Mnist(string images_path,string labels_path, string json_path);
	~Mnist();
	void MnistTest();
	void JsonTest();
	Blob* GetImages();
	Blob* GetLabels();

private:
	string images_path;
	string labels_path;
	string json_path;
	NetParam net;
	vector<string> layers;
	vector<string> ltypes;
	Blob* images;
	Blob* labels;
	int  ReverseInt(int i);
	void ReadMnistData(string path, Blob* &images);
	void ReadMnistLabel(string path, Blob* &labels);
};

#endif // __Minist_HPP__