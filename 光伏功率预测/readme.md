# 基于LSTM算法的光伏功率预测

**·**主要内容：通过历史数据构造特征输入和预测标签，将数据送入LSTM神经网络进行预测，根据损失函数不断更新权重值，构建预测模型，输出预测值并画出图形。

## 0.代码功能解释

- 根据光伏功率的历史数据预测未来4小时/24小时的光伏功率数据




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

## 2、代码运行说明

- 先调用训练函数进行模型训练，运行代码如下：


```
python LSTM_train.py
```

- 再调用验证函数，利用保存的模型进行验证，输出预测结果，画出真实值和预测值的对比图，验证模型好坏，运行代码如下： 

```
python LSTM_validation.py
```

- 最后调用主函数，从数据库读取数据，经过模型得到预测结果并输出到数据库，运行代码如下：

```
python predict_main.py
```

## 3、输入数据

- 输入：实时数据的时刻、历史光伏功率、光照强度、温度、湿度、ID等

  ![image](https://github.com/CPES-EMS/LSTM--/blob/main/%E5%85%89%E4%BC%8F%E5%8A%9F%E7%8E%87%E9%A2%84%E6%B5%8B/20231129222004.png)

  

  ![image](https://github.com/CPES-EMS/LSTM--/blob/main/%E5%85%89%E4%BC%8F%E5%8A%9F%E7%8E%87%E9%A2%84%E6%B5%8B/20231129222037.png)

  

  

## 4、输出数据

- 输出：未来24小时/4小时光伏功率，预测时间，真实时间等

![image](https://github.com/CPES-EMS/LSTM--/blob/main/%E5%85%89%E4%BC%8F%E5%8A%9F%E7%8E%87%E9%A2%84%E6%B5%8B/20231129222125.png)



![image](https://github.com/CPES-EMS/LSTM--/blob/main/%E5%85%89%E4%BC%8F%E5%8A%9F%E7%8E%87%E9%A2%84%E6%B5%8B/20231130091904.png)

## 5、代码所支撑的前端页面

![image](https://github.com/CPES-EMS/LSTM--/blob/main/%E5%85%89%E4%BC%8F%E5%8A%9F%E7%8E%87%E9%A2%84%E6%B5%8B/20231129145323.png)
