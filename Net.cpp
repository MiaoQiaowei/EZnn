#include "Net.hpp"
#include "include/json/json.h"
#include "Blob.hpp"
#include <fstream>  
#include <cassert> 
#include <memory>
using namespace std;

void NetParam::readNetParam(string file)
{
	ifstream ifs;
	ifs.open(file);
	assert(ifs.is_open());   //  断言：确保json文件正确打开
	Json::Reader reader;	 //  解析器
	Json::Value value;       //  存储器
	if (reader.parse(ifs, value))
	{
		if (!value["train"].isNull())
		{
			auto &tparam = value["train"];                             //通过引用方式，可以拿到“train”对象里面的所有元素
			this->lr = tparam["learning rate"].asDouble();             //解析成Double类型存放
			this->lr_decay = tparam["lr decay"].asDouble();
			this->optimizer = tparam["optimizer"].asString();         //解析成String类型存放
			this->momentum = tparam["momentum parameter"].asDouble();
			this->num_epochs = tparam["num epochs"].asInt();           //解析成Int类型存放
			this->use_batch = tparam["use batch"].asBool();            //解析成Bool类型存放
			this->batch_size = tparam["batch size"].asInt();
			this->eval_interval = tparam["evaluate interval"].asInt();
			this->lr_update = tparam["lr update"].asBool();
			this->snap_shot = tparam["snapshot"].asBool();
			this->snapshot_interval = tparam["snapshot interval"].asInt();
			this->fine_tune = tparam["fine tune"].asBool();
			this->preTrainModel = tparam["pre train model"].asString(); //解析成String类型存放
		}
		if (!value["net"].isNull())
		{
			auto &nparam = value["net"];                                //通过引用方式，拿到“net”数组里面的所有对象
			for (int i = 0; i < (int)nparam.size(); ++i)                //遍历“net”数组里面的所有对象
			{
				auto &ii = nparam[i];                                   //通过引用方式，拿到当前对象里面的所有元素
				this->layers.push_back(ii["name"].asString());          //往层名vector中堆叠名称      a=[]   a.append()
				this->layer_types.push_back(ii["type"].asString());          //往层类型vector中堆叠类型

				if (ii["type"].asString() == "Conv")
				{
					int num = ii["kernel num"].asInt();
					int width = ii["kernel width"].asInt();
					int height = ii["kernel height"].asInt();
					int pad = ii["pad"].asInt();
					int stride = ii["stride"].asInt();

					this->layer_params[ii["name"].asString()].conv_stride = stride;
					this->layer_params[ii["name"].asString()].conv_kernels = num;
					this->layer_params[ii["name"].asString()].conv_pad = pad;
					this->layer_params[ii["name"].asString()].conv_width = width;
					this->layer_params[ii["name"].asString()].conv_height = height;
				}
				if (ii["type"].asString() == "Pool")
				{
					int width = ii["kernel width"].asInt();
					int height = ii["kernel height"].asInt();
					int stride = ii["stride"].asInt();
					this->layer_params[ii["name"].asString()].pool_stride = stride;
					this->layer_params[ii["name"].asString()].pool_width = width;
					this->layer_params[ii["name"].asString()].pool_height = height;
				}
				if (ii["type"].asString() == "Fc")
				{
					int num = ii["kernel num"].asInt();
					this->layer_params[ii["name"].asString()].fc_kernels = num;
				}
			}
		}
	}
}


void Net::Init(NetParam &net_param, vector<shared_ptr<Blob>> &images_data, vector<shared_ptr<Blob>> &labels_data)
{
	layer_names = net_param.layers;
	layer_types = net_param.layer_types;
	for (int i = 0; i < layer_names.size(); i++)
	{
		cout << "layer = " << layer_names[i] << " type = " << layer_types[i] << endl;
	}

	images_train = images_data[0];
	labels_train = labels_data[0];
	images_val = images_data[1];
	labels_val = labels_data[1];

	for (int i = 0; i < (int)layer_names.size(); ++i)   //遍历每一层
	{
		data[layer_names[i]] = vector<shared_ptr<Blob>>(3, NULL);    //为每一层创建前向计算要用到的3个Blob
		diff[layer_names[i]] = vector<shared_ptr<Blob>>(3, NULL);      //为每一层创建反向计算要用到的3个Blob
		layers_output_shapes[layer_names[i]] = vector <int> (4);
	}

	shared_ptr<Layer> p_Layer;
	vector<int>input_shape = {
			net_param.batch_size,
			images_train->GetC(),
			images_train->GetW(),
			images_train->GetH()
	};
	//打印shape
	cout << "input size:"  << " ( " << input_shape[0] << " , " << input_shape[1] << " , " << input_shape[2] << " , " << input_shape[3] << " )" << endl;

	for (int i = 0; i < (int)layer_names.size()-1; i++)
	{
		string name= layer_names[i];
		string type=layer_types[i];

		if (type == "Conv")
		{
			p_Layer.reset(new Conv);
		}
		else if (type == "Fc")
		{
			p_Layer.reset(new Fc);
		}
		else if (type == "Pool")
		{
			p_Layer.reset(new Pool);
		}
		else if (type == "Relu")
		{
			p_Layer.reset(new Relu);
		}
		p_layers[name] = p_Layer; 
		p_Layer->Init(input_shape,data[name], net_param.layer_params[name],name);
		p_Layer->CalculateShape(input_shape, layers_output_shapes[name], net_param.layer_params[name]);
		input_shape.assign(layers_output_shapes[name].begin(), layers_output_shapes[name].end());
		cout << "Shape:";
		cout << "(" << layers_output_shapes[name][0];
		cout << "," << layers_output_shapes[name][1];
		cout << "," << layers_output_shapes[name][2];
		cout << "," << layers_output_shapes[name][3] <<")"<< endl;
	}	
}

void Net::Train(NetParam & net_param)
{
	/*从数据集中找到batch*/
	//总样本数目：
	int total_num = images_train->GetN();//一共有多少样本
	int batch_num = net_param.batch_size;//batch_size
	int per_epoch = total_num / batch_num;//单个批次中有多少组batch
	int total_batch_num = per_epoch * batch_num;//一共要训练多少组batch的数据
	cout << "total_batch_num: " << total_batch_num << endl;
	for (int i = 0; i < 40; i++)
	{
		shared_ptr<Blob>images_batch;
		shared_ptr<Blob>labels_batch;

		images_batch.reset(new Blob(images_train->SubBlob(i*batch_num % total_num, (i + 1)*batch_num % total_num)));
		labels_batch.reset(new Blob(labels_train->SubBlob(i*batch_num % total_num, (i + 1)*batch_num % total_num)));
	
		/*训练*/
		TrainWithBatch(images_batch, labels_batch, net_param);
		//cout <<"iter:  "<<i<< "  loss: " << loss << endl;

		///*参数更新*/
		///*评估当前准确率*/
		//----------step2. 用该mini-batch训练网络模型

		//----------step3. 评估模型当前准确率（训练集和验证集）
		EvaluateWithBatch(net_param);
		printf("iter_%d    lr: %0.6f    loss: %f    train_acc: %0.2f%%    val_acc: %0.2f%%\n",
			i, net_param.lr, loss, train_accu * 100, val_accu * 100);
	}
}

void Net::TrainWithBatch(shared_ptr<Blob> & images, shared_ptr<Blob> & labels, NetParam &param,string mode)
{
	/*填入X*/
	data[layer_names[0]][0] = images;
	data[layer_names.back()][1] = labels;
	
	/*逐层前项计算*/
	int n = (int)layer_names.size();
	
	for (int i = 0; i < n-1; i++)
	{
		string name = layer_names[i];
		shared_ptr<Blob>out;
		p_layers[name]->forward(data[name],out,param.layer_params[name]);
		data[layer_names[i+1]][0] = out;
	}
	if (mode == "TEST")
		return;
	
	/*计算交叉熵损失*/
	Softmax::softmax_cross_entropy_with_logits(data[layer_names.back()], loss, diff[layer_names.back()][0]);
	
	/*反向传播*/
	for (int i = n - 2; i >= 0; --i)
	{
		string name = layer_names[i];
		p_layers[name]->backward(diff[layer_names[i + 1]][0], data[name], diff[name], param.layer_params[name]);
	}

	/*参数更新*/
	OptimizerWithBatch(param);
}


void Net::OptimizerWithBatch(NetParam& param)
{
	for (auto name : layer_names)    //for lname in layers_
	{
		//(1).跳过没有w和b的层
		if (!data[name][1] || !data[name][2])
		{
			continue;  //跳过本轮循环，重新执行循环（注意不是像break那样直接跳出循环）
		}

		//(2).利用梯度下降更新有w和b的层
		for (int i = 1; i < 2; ++i)
		{
			assert(param.optimizer == "sgd" || param.optimizer == "momentum" || param.optimizer == "rmsprop");//sgd/momentum/rmsprop
			//w:=w-param.lr*dw ;    b:=b-param.lr*db     ---->  "sgd"
			shared_ptr<Blob> temp(new Blob(data[name][i]->size(), TZEROS));
			(*temp) = -param.lr * (*diff[name][i]);
			(*data[name][i]) = (*data[name][i]) + (*temp);
		}
	}
	//学习率更新
	if (param.lr_update)
		param.lr *= param.lr_decay;
}

void Net::EvaluateWithBatch(NetParam& param)
{
	//(1).评估训练集准确率
	shared_ptr<Blob> X_train_subset;
	shared_ptr<Blob> Y_train_subset;
	int N = images_train->GetN();
	if (N > 1000)
	{
		X_train_subset.reset(new Blob(images_train->SubBlob(0, 1000)));
		Y_train_subset.reset(new Blob(labels_train->SubBlob(0, 1000)));
	}
	else
	{
		X_train_subset = images_train;
		Y_train_subset = labels_train;
	}
	TrainWithBatch(X_train_subset, Y_train_subset, param, "TEST");  //“TEST”，测试模式，只进行前向传播
	train_accu_ = CalculateAccuracy(*data[layer_names.back()][1], *data[layer_names.back()][0]);

	//(2).评估验证集准确率
	TrainWithBatch(images_val, labels_val, param, "TEST");  //“TEST”，测试模式，只进行前向传播
	val_accu_ = CalculateAccuracy(*data[layer_names.back()][1], *data[layer_names.back()][0]);
}

double Net::CalculateAccuracy(Blob& in, Blob& predict)
{
	//(1). 确保两个输入Blob尺寸一样
	vector<int> size_in = in.size();
	vector<int> size_p = predict.size();
	for (int i = 0; i < 4; ++i)
	{
		assert(size_in[i] == size_p[i]);  //断言：两个输入Blob的尺寸（N,C,H,W）一样！
	}
	//(2). 遍历所有cube（样本），找出标签值Y和预测值Predict最大值所在位置进行比较，若一致，则正确个数+1
	int n_in = in.GetN();  //总样本数
	int right_cnt = 0;  //正确个数
	for (int n_ = 0; n_ < n_in; ++n_)
	{
		//参考网址：http://arma.sourceforge.net/docs.html#index_min_and_index_max_member
		if (in[n_].index_max() == predict[n_].index_max())
			right_cnt++;
	}
	return (double)right_cnt / (double)n_in;   //计算准确率，返回（准确率=正确个数/总样本数）
}