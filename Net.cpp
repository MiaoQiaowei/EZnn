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
			this->update = tparam["update method"].asString();         //������String���ʹ��
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
	for (int i = 0; i < 2; i++)
	{
		shared_ptr<Blob>images_batch;
		shared_ptr<Blob>labels_batch;

		images_batch.reset(new Blob(images_train->SubBlob(i*batch_num % total_num, (i + 1)*batch_num % total_num)));
		labels_batch.reset(new Blob(labels_train->SubBlob(i*batch_num % total_num, (i + 1)*batch_num % total_num)));
	
		/*ѵ��*/
		TrainWithBatch(images_batch, labels_batch, net_param);

		/*��������*/
		/*������ǰ׼ȷ��*/
	}
}

void Net::TrainWithBatch(shared_ptr<Blob> & images, shared_ptr<Blob> & labels, NetParam &param)
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
		cout << "name : " << name <<endl;
		p_layers[name]->forward(data[name],out,param.layer_params[name]);
		data[layer_names[i+1]][0] = out;
	}

	
	/*������ʧ*/
	Softmax::softmax_cross_entropy_with_logits(data[layer_names.back()], loss, diff[layer_names.back()][0]);
	cout << "loss: " << loss << endl;

	/*���򴫲�*/
}