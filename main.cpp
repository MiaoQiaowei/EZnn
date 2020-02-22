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
	string imagesPath = "D:\\github\\EZnn\\Mnist\\train\\train-images.idx3-ubyte";
	string labelsPath = "D:\\github\\EZnn\\Mnist\\train\\train-labels.idx1-ubyte";
	Mnist mnist(imagesPath, labelsPath, ModelPath);
	mnist.Train();
}