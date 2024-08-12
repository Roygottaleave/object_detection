# RetinaNet

## 该项目主要是来自pytorch官方torchvision模块中的源码
* https://github.com/pytorch/vision/tree/master/torchvision/models/detection


## 文件结构：
```
  ├── backbone: 特征提取网络(ResNet50+FPN)
  ├── network_files: RetinaNet网络
  ├── train_utils: 训练验证相关模块（包括cocotools）
  ├── my_dataset.py: 自定义dataset用于读取VOC数据集
  ├── train.py: 以resnet50+FPN做为backbone进行训练
  ├── train_multi_GPU.py: 针对使用多GPU的用户使用
  ├── predict.py: 简易的预测脚本，使用训练好的权重进行预测测试
  ├── validation.py: 利用训练好的权重验证/测试数据的COCO指标，并生成record_mAP.txt文件
  └── pascal_voc_classes.json: pascal_voc标签文件(注意索引从0开始，不包括背景)
```

## 预训练权重下载地址（下载后放入backbone文件夹中）：
* ResNet50+FPN backbone: https://download.pytorch.org/models/retinanet_resnet50_fpn_coco-eeacb38b.pth
* 注意，下载的预训练权重记得要重命名，比如在train.py中读取的是`retinanet_resnet50_fpn_coco.pth`文件，
  不是`retinanet_resnet50_fpn_coco-eeacb38b.pth`


## 训练方法
* 确保提前准备好数据集
* 确保提前下载好对应预训练模型权重
* 若要单GPU训练，直接使用train.py训练脚本
* 若要使用多GPU训练，使用`python -m torch.distributed.launch --nproc_per_node=8 --use_env train_multi_GPU.py`指令,`nproc_per_node`参数为使用GPU数量
* 如果想指定使用哪些GPU设备可在指令前加上`CUDA_VISIBLE_DEVICES=0,3`
* `CUDA_VISIBLE_DEVICES=0,3 python -m torch.distributed.launch --nproc_per_node=2 --use_env train_multi_GPU.py`


