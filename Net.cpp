#include "Net.hpp"
#include "include/json/json.h"
#include <fstream>  
#include <cassert>  
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
			this->update = tparam["update method"].asString();         //解析成String类型存放
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
				this->ltypes.push_back(ii["type"].asString());          //往层类型vector中堆叠类型

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