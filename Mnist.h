#ifndef  __Mnist_HPP__
#define __Mnist_HPP__
#include "Net.hpp"
#include "Blob.hpp"
#include <iostream>
#include <string>
#include <memory>
#include "Net.hpp"


using std::string;

class Mnist
{
public:
	Mnist(string images_path, string labels_path, string json_path);
	~Mnist();
	void MnistTest();
	void JsonTest();
	Blob* GetImages();
	Blob* GetLabels();
	void Train(string config_file, shared_ptr<Blob> images, shared_ptr<Blob> labels);
	void Train();

private:
	string images_path;
	string labels_path;
	string json_path;
	NetParam net_param;
	vector<string> layers;
	vector<string> layer_types;
	Blob* images;
	Blob* labels;
	int  ReverseInt(int i);
	void ReadMnistData(string path, Blob* &images);
	void ReadMnistLabel(string path, Blob* &labels);
};

#endif // __Minist_HPP__