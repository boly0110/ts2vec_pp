# Linux端基础训练预测功能测试

Linux端基础训练预测功能测试的主程序为`test_train_inference_python.sh`，可以测试基于Python的模型训练、评估、推理等基本功能，包括裁剪、量化、蒸馏。

## 1. 测试结论汇总

- 训练相关：

| 算法名称 | 模型名称   | 单机单卡 |
|  :----  |:-------|    :----  |
|  CNN  | ts2vec | 正常训练 |


- 预测相关：基于训练是否使用量化，可以将训练产出的模型可以分为`正常模型`和`量化模型`，这两类模型对应的预测功能汇总如下，

| 模型类型 |device | batchsize | tensorrt  | mkldnn | cpu多线程 |
|  ----   |  ---- |   ----   |:---------:|   :----:   |  :----:  |
| 正常模型 | GPU | 1 | fp32/fp16 | - | - |
| 正常模型 | CPU | 1 |     -     | - | 支持 |


## 2. 测试流程


### 2.1 安装依赖
- 安装PaddlePaddle == 2.3.1

- 安装z支持包
    ```
    pip install  -r ../requirements.txt
    ```


### 2.2 功能测试
先运行`prepare.sh`准备数据和模型，然后运行`test_train_inference_python.sh`进行测试，最终在```test_tipc/output```目录下生成`python_infer_*.log`格式的日志文件。

`test_train_inference_python.sh`用于测试速度和精度：

```shell
bash test_tipc/prepare.sh ./test_tipc/configs/paconv/train_infer_python.txt 'lite_train_lite_infer'
bash test_tipc/test_train_inference_python.sh ./test_tipc/configs/paconv/train_infer_python.txt 'lite_train_lite_infer'
```
