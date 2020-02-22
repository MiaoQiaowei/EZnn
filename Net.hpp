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

struct NetParam      //c++中，struct跟class用法基本一致！主要区别是继承后的数据访问权限。
{
	/*学习率*/
	double lr;
	/*学习率衰减系数*/
	double lr_decay;
	/*优化算法,:sgd/momentum/rmsprop*/
	std::string update;
	/*momentum系数 */
	double momentum;
	/*epoch次数 */
	int num_epochs;
	/*是否使用mini-batch梯度下降*/
	bool use_batch;
	/*每批次样本个数*/
	int batch_size;
	/*每隔几个迭代周期评估一次准确率？ */
	int eval_interval;
	/*是否更新学习率？  true/false*/
	bool lr_update;
	/* 是否保存模型快照；快照保存间隔*/
	bool snap_shot;
	/*每隔几个迭代周期保存一次快照？*/
	int snapshot_interval;
	/* 是否采用fine-tune方式训练*/
	bool fine_tune;
	/*预训练模型文件.gordonmodel所在路径*/
	string preTrainModel;
	/*层名*/
	vector <string> layers;
	/*层类型*/
	vector <string> layer_types;
	/*无序关联容器*/
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