#include "Net.hpp"
#include "include/json/json.h"
#include <fstream>  
#include <cassert>  
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
				this->ltypes.push_back(ii["type"].asString());          //��������vector�жѵ�����

				if (ii["type"].asString() == "Conv")
				{
					int num = ii["kernel num"].asInt();
					int width = ii["kernel width"].asInt();
					int height = ii["kernel height"].asInt();
					int pad = ii["pad"].asInt();
					int stride = ii["stride"].asInt();

					this->lparams[ii["name"].asString()].conv_stride = stride;
					this->lparams[ii["name"].asString()].conv_kernels = num;
					this->lparams[ii["name"].asString()].conv_pad = pad;
					this->lparams[ii["name"].asString()].conv_width = width;
					this->lparams[ii["name"].asString()].conv_height = height;
				}
				if (ii["type"].asString() == "Pool")
				{
					int width = ii["kernel width"].asInt();
					int height = ii["kernel height"].asInt();
					int stride = ii["stride"].asInt();
					this->lparams[ii["name"].asString()].pool_stride = stride;
					this->lparams[ii["name"].asString()].pool_width = width;
					this->lparams[ii["name"].asString()].pool_height = height;
				}
				if (ii["type"].asString() == "Fc")
				{
					int num = ii["kernel num"].asInt();
					this->lparams[ii["name"].asString()].fc_kernels = num;
				}
			}
		}


	}


}