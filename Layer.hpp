#ifndef  __LAYER_HPP__
#define __LAYER_HPP__
#include <iostream>

struct Param  //�ṹ�� �����������Ҫ��ÿһ���ϸ�ڲ��֣�
{
	/*1.����㳬���� */
	int conv_stride;
	int conv_pad;
	int conv_width;
	int conv_height;
	int conv_kernels;

	/*2.�ػ��㳬����*/
	int pool_stride;
	int pool_width;
	int pool_height;

	/*3.ȫ���Ӳ㳬���������ò���Ԫ������ */
	int fc_kernels;
};

class Layer
{
public:
	//Layer();
	//virtual ~Layer();
	virtual void Init() = 0;

private:

};


class Conv:public Layer
{
public:
	void Init();

private:

};

class Pool :public Layer
{
public:
	void Init();

private:

};
class Relu :public Layer
{
public:
	void Init();

private:

};

class Fc :public Layer
{
public:
	void Init();

private:

};




#endif  //__LAYER_HPP__