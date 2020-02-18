#ifndef __MYNET_HPP__
#define __MYNET_HPP__
#include "myLayer.hpp"
#include <iostream>
#include <vector>
#include <unordered_map>

using std::unordered_map;
using std::vector;
using std::string;


struct NetParam      //c++�У�struct��class�÷�����һ�£���Ҫ�����Ǽ̳к�����ݷ���Ȩ�ޡ�
{
	/*ѧϰ��*/
	double lr;
	/*ѧϰ��˥��ϵ��*/
	double lr_decay;
	/*�Ż��㷨,:sgd/momentum/rmsprop*/
	std::string update;
	/*momentumϵ�� */
	double momentum;
	/*epoch���� */
	int num_epochs;
	/*�Ƿ�ʹ��mini-batch�ݶ��½�*/
	bool use_batch;
	/*ÿ������������*/
	int batch_size;
	/*ÿ������������������һ��׼ȷ�ʣ� */
	int eval_interval;
	/*�Ƿ����ѧϰ�ʣ�  true/false*/
	bool lr_update;
	/* �Ƿ񱣴�ģ�Ϳ��գ����ձ�����*/
	bool snap_shot;
	/*ÿ�������������ڱ���һ�ο��գ�*/
	int snapshot_interval;
	/* �Ƿ����fine-tune��ʽѵ��*/
	bool fine_tune;
	/*Ԥѵ��ģ���ļ�.gordonmodel����·��*/
	string preTrainModel;

	/*����*/
	vector <string> layers;
	/*������*/
	vector <string> ltypes;

	/*�����������*/
	unordered_map<string, Param> lparams;


	void readNetParam(string file);


};

#endif