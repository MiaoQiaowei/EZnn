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
	assert(ifs.is_open());   //  ���ԣ�ȷ��json�ļ���ȷ��
	Json::Reader reader;	 //  ������
	Json::Value value;       //  �洢��
	if (reader.parse(ifs, value))
	{
		if (!value["train"].isNull())
		{
			auto &tparam = value["train"];                             //ͨ�����÷�ʽ�������õ���train���������������Ԫ��
			this->lr = tparam["learning rate"].asDouble();             //������Double���ʹ��
			this->lr_decay = tparam["lr decay"].asDouble();
			this->optimizer = tparam["optimizer"].asString();         //������String���ʹ��
			this->momentum = tparam["momentum parameter"].asDouble();
			this->num_epochs = tparam["num epochs"].asInt();           //������Int���ʹ��
			this->use_batch = tparam["use batch"].asBool();            //������Bool���ʹ��
			this->batch_size = tparam["batch size"].asInt();
			this->eval_interval = tparam["evaluate interval"].asInt();
			this->lr_update = tparam["lr update"].asBool();
			this->snap_shot = tparam["snapshot"].asBool();
			this->snapshot_interval = tparam["snapshot interval"].asInt();
			this->fine_tune = tparam["fine tune"].asBool();
			this->preTrainModel = tparam["pre train model"].asString(); //������String���ʹ��
		}
		if (!value["net"].isNull())
		{
			auto &nparam = value["net"];                                //ͨ�����÷�ʽ���õ���net��������������ж���
			for (int i = 0; i < (int)nparam.size(); ++i)                //������net��������������ж���
			{
				auto &ii = nparam[i];                                   //ͨ�����÷�ʽ���õ���ǰ�������������Ԫ��
				this->layers.push_back(ii["name"].asString());          //������vector�жѵ�����      a=[]   a.append()
				this->layer_types.push_back(ii["type"].asString());          //��������vector�жѵ�����

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

	for (int i = 0; i < (int)layer_names.size(); ++i)   //����ÿһ��
	{
		data[layer_names[i]] = vector<shared_ptr<Blob>>(3, NULL);    //Ϊÿһ�㴴��ǰ�����Ҫ�õ���3��Blob
		diff[layer_names[i]] = vector<shared_ptr<Blob>>(3, NULL);      //Ϊÿһ�㴴���������Ҫ�õ���3��Blob
		layers_output_shapes[layer_names[i]] = vector <int> (4);
	}

	shared_ptr<Layer> p_Layer;
	vector<int>input_shape = {
			net_param.batch_size,
			images_train->GetC(),
			images_train->GetW(),
			images_train->GetH()
	};
	//��ӡshape
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
	/*�����ݼ����ҵ�batch*/
	//��������Ŀ��
	int total_num = images_train->GetN();//һ���ж�������
	int batch_num = net_param.batch_size;//batch_size
	int per_epoch = total_num / batch_num;//�����������ж�����batch
	int total_batch_num = per_epoch * batch_num;//һ��Ҫѵ��������batch������
	cout << "total_batch_num: " << total_batch_num << endl;
	for (int i = 0; i < 40; i++)
	{
		shared_ptr<Blob>images_batch;
		shared_ptr<Blob>labels_batch;

		images_batch.reset(new Blob(images_train->SubBlob(i*batch_num % total_num, (i + 1)*batch_num % total_num)));
		labels_batch.reset(new Blob(labels_train->SubBlob(i*batch_num % total_num, (i + 1)*batch_num % total_num)));
	
		/*ѵ��*/
		TrainWithBatch(images_batch, labels_batch, net_param);
		//cout <<"iter:  "<<i<< "  loss: " << loss << endl;

		///*��������*/
		///*������ǰ׼ȷ��*/
		//----------step2. �ø�mini-batchѵ������ģ��

		//----------step3. ����ģ�͵�ǰ׼ȷ�ʣ�ѵ��������֤����
		EvaluateWithBatch(net_param);
		printf("iter_%d    lr: %0.6f    loss: %f    train_acc: %0.2f%%    val_acc: %0.2f%%\n",
			i, net_param.lr, loss, train_accu * 100, val_accu * 100);
	}
}

void Net::TrainWithBatch(shared_ptr<Blob> & images, shared_ptr<Blob> & labels, NetParam &param,string mode)
{
	/*����X*/
	data[layer_names[0]][0] = images;
	data[layer_names.back()][1] = labels;
	
	/*���ǰ�����*/
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
	
	/*���㽻������ʧ*/
	Softmax::softmax_cross_entropy_with_logits(data[layer_names.back()], loss, diff[layer_names.back()][0]);
	
	/*���򴫲�*/
	for (int i = n - 2; i >= 0; --i)
	{
		string name = layer_names[i];
		p_layers[name]->backward(diff[layer_names[i + 1]][0], data[name], diff[name], param.layer_params[name]);
	}

	/*��������*/
	OptimizerWithBatch(param);
}


void Net::OptimizerWithBatch(NetParam& param)
{
	for (auto name : layer_names)    //for lname in layers_
	{
		//(1).����û��w��b�Ĳ�
		if (!data[name][1] || !data[name][2])
		{
			continue;  //��������ѭ��������ִ��ѭ����ע�ⲻ����break����ֱ������ѭ����
		}

		//(2).�����ݶ��½�������w��b�Ĳ�
		for (int i = 1; i < 2; ++i)
		{
			assert(param.optimizer == "sgd" || param.optimizer == "momentum" || param.optimizer == "rmsprop");//sgd/momentum/rmsprop
			//w:=w-param.lr*dw ;    b:=b-param.lr*db     ---->  "sgd"
			shared_ptr<Blob> temp(new Blob(data[name][i]->size(), TZEROS));
			(*temp) = -param.lr * (*diff[name][i]);
			(*data[name][i]) = (*data[name][i]) + (*temp);
		}
	}
	//ѧϰ�ʸ���
	if (param.lr_update)
		param.lr *= param.lr_decay;
}

void Net::EvaluateWithBatch(NetParam& param)
{
	//(1).����ѵ����׼ȷ��
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
	TrainWithBatch(X_train_subset, Y_train_subset, param, "TEST");  //��TEST��������ģʽ��ֻ����ǰ�򴫲�
	train_accu_ = CalculateAccuracy(*data[layer_names.back()][1], *data[layer_names.back()][0]);

	//(2).������֤��׼ȷ��
	TrainWithBatch(images_val, labels_val, param, "TEST");  //��TEST��������ģʽ��ֻ����ǰ�򴫲�
	val_accu_ = CalculateAccuracy(*data[layer_names.back()][1], *data[layer_names.back()][0]);
}

double Net::CalculateAccuracy(Blob& in, Blob& predict)
{
	//(1). ȷ����������Blob�ߴ�һ��
	vector<int> size_in = in.size();
	vector<int> size_p = predict.size();
	for (int i = 0; i < 4; ++i)
	{
		assert(size_in[i] == size_p[i]);  //���ԣ���������Blob�ĳߴ磨N,C,H,W��һ����
	}
	//(2). ��������cube�����������ҳ���ǩֵY��Ԥ��ֵPredict���ֵ����λ�ý��бȽϣ���һ�£�����ȷ����+1
	int n_in = in.GetN();  //��������
	int right_cnt = 0;  //��ȷ����
	for (int n_ = 0; n_ < n_in; ++n_)
	{
		//�ο���ַ��http://arma.sourceforge.net/docs.html#index_min_and_index_max_member
		if (in[n_].index_max() == predict[n_].index_max())
			right_cnt++;
	}
	return (double)right_cnt / (double)n_in;   //����׼ȷ�ʣ����أ�׼ȷ��=��ȷ����/����������
}