#ifndef  __Mnist_HPP__
#define __Mnist_HPP__
#include <iostream>
#include <string>
#include <memory>
#include "Net.hpp"
#include "Blob.hpp"
#include "Net.hpp"


using std::string;
using std::shared_ptr;

class Mnist
{
public:
	Mnist(string images_path, string labels_path, string json_path);
	shared_ptr<Blob> GetImages();
	shared_ptr<Blob> GetLabels();
	void MnistTest();
	void JsonTest();
	void Train(string config_file, shared_ptr<Blob> images, shared_ptr<Blob> labels);
	void Train();

private:

	NetParam net_param;
	string images_path;
	string labels_path;
	string json_path;
	shared_ptr<Blob> images;
	shared_ptr<Blob> labels;
	vector<string> layers;
	vector<string> layer_types;
	
	int  ReverseInt(int i);
	void ReadMnistData(string path, shared_ptr<Blob> &images);
	void ReadMnistLabel(string path, shared_ptr<Blob> &labels);
};

#endif // __Minist_HPP__