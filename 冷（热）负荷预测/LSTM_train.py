import datetime
import os  # 导入 os 库，用于处理文件路径
import math
import traceback
from config import logging, window_input_size, window_label_size, batch_size, input_size, hidden_size, output_size, \
    device, epoch
import pandas as pd  # 导入 Pandas 库，用于数据处理
import torch  # 导入 PyTorch 库
import torch.nn as nn  # 导入 PyTorch 库中的神经网络模块，用于定义损失函数等
from torch.utils.data import Dataset, DataLoader  # 从 PyTorch 库中导入 Dataset 和 DataLoader 类，用于构建自定义数据集和数据加载器
from model import LstmNet  # 导入自定义的模型类 LstmNet
from torch.optim.lr_scheduler import CosineAnnealingLR
from dateutil import parser

""" 自定义数据集类，用于构建 PyTorch 的数据加载器（DataLoader），用于训练或评估模型
 Args:
      window_input_size: 输入窗口
      window_label_size: 输出窗口
      window_gap: 滑动步长
      start_index: 起始索引
      end_index: 结束索引
      inputs:输入
      labels:标签"""


class MyDataset(Dataset):
    def __init__(self, data, window_input_size, window_label_size):
        self.data = pd.DataFrame(data)  # 将 data 转换为 Pandas 数据框
        self.window_input_size = window_input_size  # 输入窗口大小
        self.window_label_size = window_label_size  # 输出窗口大小
        self.window_gap = 24  # 滑动步长

    def __getitem__(self, index):  # 获取数据
        start_index = self.window_gap * index  # 计算起始索引
        end_index = start_index + self.window_input_size + self.window_label_size  # 计算结束索引
        window_input = torch.tensor(  # 构造输入 Tensor
            self.data.iloc[start_index: self.window_input_size + start_index, 1:].values,
            dtype=torch.float32,  # 数据类型为 float32
        )
        window_label = torch.tensor(
            self.data.iloc[self.window_input_size + start_index: end_index, 4:5].values,  # 将DataFrame转换为NumPy数组
            dtype=torch.float32,
        )
        inputs = window_input  # 输入
        label = window_label  # 标签
        return inputs, label

    def __len__(self):  # 获取数据长度
        # 返回窗口个数，即数据总长度除以输入窗口大小减 1
        num_windows = (len(self.data) - (self.window_input_size + self.window_label_size)) // self.window_gap + 1
        return math.ceil(num_windows)


if __name__ == "__main__":
    # 读取数据
    filedir = "./data/pv"  # 数据文件夹路径
    filename = "yulin_load.xlsx"  # 数据文件名
    try:
        logging.info("读入数据")
        filepath = os.path.join(filedir, filename)  # 构造数据文件完整路径
        data = pd.read_excel(filepath)  # 读取 CSV 格式的数据文件
    except Exception as e:
        logging.error("读取数据失败, 失败原因为")
        logging.error(traceback.format_exc())
        raise e
    logging.info("数据读取成功")
    try:
        logging.info("时间列转换")
        data['日期'] = pd.to_datetime(data['日期'], format='%Y-%m-%d %H:%M:%S')
    except Exception as e:
        logging.error("时间列不存在")
        logging.error(traceback.format_exc())
        raise e
    logging.info("时间列转换成功")
    # 拆分时间列为年、月、日、小时、分钟四列
    data['year'] = data['日期'].dt.year
    data['month'] = data['日期'].dt.month
    data['day'] = data['日期'].dt.day
    data['hour'] = data['日期'].dt.hour
    # 增添小时列
    hour = 0
    for row in data.itertuples():
        data.at[row.Index, 'hour'] = hour
        hour = (hour + 1) % 24
    week_list = []
    # 增添时间列为星期几
    for row in range(data.shape[0]):
        weekdays = data['日期'][row]
        week_list.append(weekdays.weekday())
    data['weekdays'] = pd.DataFrame(week_list)
    # 对周末进行标记，热负荷可能变化
    data.loc[:, 'weekend'] = 0
    data.loc[:, 'weekend_sat'] = 0
    data.loc[:, 'weekend_sun'] = 0
    data.loc[(data['weekdays'] > 4), 'weekend'] = 1
    data.loc[(data['weekdays'] == 5), 'weekend_sat'] = 1
    data.loc[(data['weekdays'] == 6), 'weekend_sun'] = 1
    # 夏天冷负荷，冬天热负荷，对夏天冬天进行标记
    month_list = []
    for row in range(data.shape[0]):
        months = data['month'][row]
        print(type(month_list))
        month_list.append(months)
    data['win_or_sum'] = pd.DataFrame(month_list)
    print(type(data['win_or_sum'].values))
    data.loc[:, 'summer'] = 0
    data.loc[:, 'winter'] = 0
    data.loc[(6 <= data['win_or_sum']), 'summer'] = 1  # 6月-9月定义为夏天
    data.loc[(data['win_or_sum'] >= 10), 'summer'] = 0
    data.loc[(data['win_or_sum'] >= 11), 'winter'] = 1  # 11月到次年3月定义为冬天
    data.loc[(data['win_or_sum'] <= 3), 'winter'] = 1
    print(data)
    # # 删除原来的时间列
    # data = data.drop(['时间'], axis=1)
    try:
        # 调整列的顺序，将新列插入到合适的位置
        new_order = ['year', 'month', 'day', 'hour', '供暖热负荷(kW)', '电负荷kW', 'weekend', 'weekend_sat',
                     'winter', 'weekend_sun','summer']
         # 将"..."替换为原来的列名及其顺序
        data = data[new_order]
    except Exception as e:
        logging.error("数据修改失败")
        logging.error(traceback.format_exc())
        raise e
    print(data)
    try:
        logging.info("存储新的数据集")
        # 将处理后的数据写入到新的的CSV文件中, 不写入索引
        data.to_csv('./handled_data/yulin_load_elc.csv', index=False)
    except Exception as e:
        logging.error("数据存储失败,失败原因为")
        logging.error(traceback.format_exc())
        raise e
    logging.info("新数据集存储成功")
    # 划分训练集和验证集
    train_data = data[: int(len(data) * 0.80)]  # 前 80% 的数据作为训练集
    try:
        logging.info("构建训练数据集")
        # 创建训练集和验证集的数据集和数据加载器
        Lstm_train_dataset = MyDataset(train_data, window_input_size, window_label_size)  # 构造训练数据集
        Lstm_train_dataloader = DataLoader(Lstm_train_dataset, batch_size, shuffle=False)  # 构造训练数据加载器，不进行随机打乱操作
    except Exception as e:
        logging.error("构建失败，失败原因为")
        logging.error(traceback.format_exc())
        raise e
    logging.info("构建成功")

    # 实例化模型
    Lstm_Net = LstmNet(input_size, hidden_size, output_size, window_label_size).to(device)  # 构造 LSTM 模型，将模型放到指定设备上
    # 设置优化器和损失函数
    optimizer = torch.optim.AdamW(Lstm_Net.parameters(), lr=0.001, weight_decay=0.01)  # 使用 AdamW 优化器
    # 使用smooth误差损失函数
    criterion = nn.SmoothL1Loss()
    # 定义余弦退火学习率调整器
    scheduler = CosineAnnealingLR(optimizer, T_max=100, eta_min=0.001)

    # 初始化列表来存储训练和验证损失
    train_losses = []  # 训练损失列表
    for i in range(epoch):  # 开始训练
        # 开始训练
        Lstm_Net.train()  # 设置为训练模式
        train_loss = 0
        try:
            for batch_num, (x, label) in enumerate(Lstm_train_dataloader):  # 遍历训练数据加载器
                x = x.clone().detach().requires_grad_(True).to(device)  # 将输入放到指定设备上
                label = label.clone().detach().requires_grad_(True).to(device)  # 将标签放到指定设备上
                output = Lstm_Net(x)  # 模型前向传递
                output = output[:, -window_label_size:, :]  # 取输出序列中最后的 window_label_size 个时间步
                loss = criterion(output, label)  # 计算损失
                train_loss += loss.item()
                optimizer.zero_grad()  # 清空梯度
                loss.backward()  # 反向传播计算梯度
                # 设置梯度阈值
                torch.nn.utils.clip_grad_norm_(Lstm_Net.parameters(), max_norm=0.01)
                optimizer.step()  # 更新模型参数
        except Exception as e:
            logging.error("训练失败，失败原因为")
            logging.error(traceback.format_exc())
            raise e
        # 在每个epoch结束时更新学习率
        scheduler.step()
        # 记录并打印训练损失
        train_losses.append(train_loss / (batch_num + 1))  # 存储训练损失
        print("Train Epoch: %d, Loss: %.4f" % (i, train_loss / (batch_num + 1)))  # 打印训练损失

    try:
        logging.info("保存模型参数")
        # 将模型的参数保存到文件中，文件名为 model.pt
        torch.save(Lstm_Net.state_dict(), "./output_results/model_4h1.pt")
    except Exception as e:
        logging.error("保存失败，失败原因为")
        logging.error(traceback.format_exc())
        raise e
    logging.info("模型保存成功")
