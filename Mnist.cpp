#include "Mnist.h"
#include <iostream>
#include <string>
#include <memory>

using namespace std;

Mnist::Mnist(string imagesPath, string labelsPath, string JsonPath)
{
	this->imagesPath = imagesPath;
	this->labelsPath = labelsPath;
	this->JsonPath = JsonPath;
}

Mnist::~Mnist()
{
	delete(this->images);
	delete(this->labels);
}

int Mnist::ReverseInt(int i)  ////�Ѵ������ת��Ϊ���ǳ��õ�С������ ����С��ģʽ��ԭ��
{
	unsigned char ch1, ch2, ch3, ch4;  //һ��int��4��char
	ch1 = i & 255;
	ch2 = (i >> 8) & 255;
	ch3 = (i >> 16) & 255;
	ch4 = (i >> 24) & 255;
	return ((int)ch1 << 24) + ((int)ch2 << 16) + ((int)ch3 << 8) + ch4;
}

void Mnist::ReadMnistData(string path, Blob* &images)
{
	ifstream file(path, ios::binary);
	if (file.is_open())
	{
		//mnistԭʼ�����ļ���32λ������ֵ�Ǵ�˴洢��C/C++������С�˴洢�����Զ�ȡ���ݵ�ʱ����Ҫ������д�С��ת��!!!!
		//1.���ļ��л�֪ħ�����֣�һ�㶼���𵽱�ʶ�����ã����������ж�����ļ��ǲ���MNIST�����train-labels.idx1-ubyte�ļ�����ͼƬ������ͼƬ������Ϣ
		int magic_number = 0;
		int number_of_images = 0;
		int n_rows = 0;
		int n_cols = 0;
		file.read((char*)&magic_number, sizeof(magic_number));
		magic_number = Mnist::ReverseInt(magic_number);  //�ߵ��ֽڵ���
		cout << "magic_number=" << magic_number << endl;
		file.read((char*)&number_of_images, sizeof(number_of_images));
		number_of_images = Mnist::ReverseInt(number_of_images);
		cout << "number_of_images=" << number_of_images << endl;
		file.read((char*)&n_rows, sizeof(n_rows));
		n_rows = Mnist::ReverseInt(n_rows);
		cout << "n_rows=" << n_rows << endl;
		file.read((char*)&n_cols, sizeof(n_cols));
		n_cols = Mnist::ReverseInt(n_cols);
		cout << "n_cols=" << n_cols << endl;

		//2.��ͼƬתΪBlob�洢��
		for (int i = 0; i < number_of_images; ++i)  //��������ͼƬ
		{
			for (int h = 0; h < n_rows; ++h)   //������
			{
				for (int w = 0; w < n_cols; ++w)   //������
				{
					unsigned char temp = 0;
					file.read((char*)&temp, sizeof(temp));      //����һ������ֵ��		
					//-----��temp�е�����д��Blob��------
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
void Mnist::ReadMnistLabel(string path, Blob* &labels)
{

	ifstream file(path, ios::binary);
	if (file.is_open())
	{
		//1.���ļ��л�֪ħ�����֣�ͼƬ����
		int magic_number = 0;
		int number_of_images = 0;
		file.read((char*)&magic_number, sizeof(magic_number));
		magic_number = Mnist::ReverseInt(magic_number);
		cout << "magic_number=" << magic_number << endl;
		file.read((char*)&number_of_images, sizeof(number_of_images));
		number_of_images = Mnist::ReverseInt(number_of_images);
		cout << "number_of_Labels=" << number_of_images << endl;
		//2.�����б�ǩתΪBlob�洢������д����ʶ��0~9��
		for (int i = 0; i < number_of_images; ++i)
		{
			unsigned char temp = 0;
			file.read((char*)&temp, sizeof(temp));
			//-----��temp�е�����д��Blob��------
			(*labels)[i](0, 0, (int)temp) = 1;

			//-----------------------------------
		}
	}
	else
	{
		cout << "no label file found :-(" << endl;
	}
}

void Mnist::MnistTest()
{
	//ʵ��������Blob����
	Blob* images(new Blob(60000, 1, 28, 28, TONES));
	Blob* labels(new Blob(60000, 10, 1, 1, TONES));
	//����·��
	ReadMnistData(this->imagesPath, images);
	ReadMnistLabel(this->labelsPath, labels);

	this->images = images;
	this->labels = labels;
}

void Mnist::JsonTest()
{
	this->net.readNetParam(this->JsonPath);
	this->layers_ = this->net.layers;
	this->ltypes_ = this->net.ltypes;
	cout << "Json info:" << endl;
	for (int i = 0; i < layers_.size(); ++i)
	{
		cout << "layer = " << layers_[i] << " ; " << "ltype = " << ltypes_[i] << endl;
	}
	cout << "Json is ok!" << endl;
}