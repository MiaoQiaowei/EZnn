# EZnn

EZnn是自己开发的一套深度学习框架。底层由C++实现，将逐步完善后打包引入Python中进行使用。主体架构类似Caffe，结构和数据双流结构。JSON完成网络结构搭建，Blob完成数据流传送。

作者：苗乔伟 Max，李佳瑞Jiarui Li。

## 更新说明：

2020-02-23

EZnn  v1.0.0 已经上传，安装好配置之后可以使用VS 选择x64 模式下进行测试。但在训练之前请务必在main.c文件中更改Mnist文件夹下数据集的路径！！！如有BUG，欢迎反馈。

测试结果：

![result](D:\Documents\Desktop\result.png)

## 	 进度

- [x] 利用JSON定义网络结构
- [x] 设计Blob结构
- [x] 加载Minist数据集进行测试
- [x] 构造数据和梯度流
- [x] 逐层初始化
- [x] Blob切割成Batch
- [x] 前向传播
- [x] 反向传播
- [x] 模型参数优化
- [x] 优化模型
- [ ] 自动求导
- [ ] 编译成静态库
- [ ] 支持GPU训练
- [ ] 设计图形界面

## 配置需要：

VS2013（如果使用更新的VS，请在安装新版本VS之后再安装VS2013，以确保包含v120工作集）

JSON

Armadillo

Protobuf 2.5.0

OpenCV

## 数据集：

Mnist数据集：

链接：https://pan.baidu.com/s/1VRM0Z-H-9psr4Nob19_UAA    提取码：4rvg 

## 联系：

Email: QiaoweiMiao@gmail.com

QQ: 252544058



