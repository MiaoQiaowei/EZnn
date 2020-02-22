#ifndef __NET_HPP__
#define __NET_HPP__
#include "Layer.hpp"
#include <iostream>
#include <vector>
#include <unordered_map>
#include <memory>
#include "Blob.hpp"

using std::unordered_map;
using std::vector;
using std::string;
using std::shared_ptr;

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
	vector <string> layer_types;
	/*�����������*/
	unordered_map<string, LayerParam> layer_params;
	void readNetParam(string file);
};

class Net
{
public:
	Net(){};
	void Init(NetParam &net, vector<shared_ptr<Blob>> &train, vector<shared_ptr<Blob>> &val);
	void Train(NetParam &net_param);
	void TrainWithBatch(shared_ptr<Blob> & images, shared_ptr<Blob> & labels, NetParam &param);

private:
	shared_ptr<Blob>images_train;
	shared_ptr<Blob>images_val;
	shared_ptr<Blob>labels_train;
	shared_ptr<Blob>labels_val;

	vector<string>layer_names;
	vector<string>layer_types;

	unordered_map<string, vector<shared_ptr<Blob>>>data;
	unordered_map<string, vector<shared_ptr<Blob>>>diff;
	unordered_map<string, shared_ptr<Layer>>p_layers;
	unordered_map<string, vector<int>>layers_output_shapes;
};

#endif