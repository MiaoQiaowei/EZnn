#ifndef __MYNET_HPP__
#define __MYNET_HPP__
#include "myLayer.hpp"
#include <iostream>
#include <vector>
#include <unordered_map>

using std::unordered_map;
using std::vector;
using std::string;


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
	vector <string> ltypes;

	/*无序关联容器*/
	unordered_map<string, Param> lparams;


	void readNetParam(string file);


};

#endif