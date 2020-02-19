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





#endif  //__LAYER_HPP__