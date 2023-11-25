# 基于LSTM算法的光伏功率预测与负荷预测

**·**主要内容：通过历史数据构造特征输入和预测标签，将数据送入LSTM神经网络进行预测，根据损失函数不断更新权重值，构建预测模型，输出预测值并画出图形。

## 0.代码功能解释

- 根据光伏功率的历史数据预测未来4小时/24小时的光伏功率数据

- 根据建筑负荷的历史数据预测未来4小时/24小时的建筑负荷数据


## 1.代码环境

```
pandas==2.0.3
numpy==1.24.3
matplotlib==3.7.2
openpyxl==3.0.10
scikit-learn==1.3.0
torch==2.0.1
pytorch-cuda==11.7
pytorch-mutex==1.0
torchaudio==2.0.2
torchvision== 0.15.2
mpmath==1.3.0
```

- 搭环境的话请按照目录下的requirements.txt 进行安装

​                ```		pip install -r requirements.txt```

## 2、输入数据

- 输入：实时数据的时刻、历史光伏功率、光照强度、温度、湿度、time_cos、time_sin

## 3、输出数据

- 输出：未来24小时/4小时光伏功率
