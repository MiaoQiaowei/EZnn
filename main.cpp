#include <iostream>
#include <string>
#include <memory>
#include "Net.hpp"
#include "Blob.hpp"
#include "Mnist.h"

using namespace std;

int main(int argc, char** argv)
{
	string ModelPath = "./Model.json";
	string imagesPath = "D:\\SOFTWARE\\github\\EZnn\\Mnist\\train\\train-images.idx3-ubyte";
	string labelsPath = "D:\\SOFTWARE\\github\\EZnn\\Mnist\\train\\train-labels.idx1-ubyte";
	Mnist mnist(imagesPath, labelsPath, ModelPath);
	cout << "--------------------------begin to train---------------------------" << endl;
	mnist.Train();
	cout << "----------------------------train end------------------------------" << endl;
}