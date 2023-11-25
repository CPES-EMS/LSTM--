
import os  # 导入 os 库，用于处理文件路径
import math
import traceback
import time
from log_print import TNLog
import matplotlib.pyplot as plt  # 导入 Matplotlib 库，用于可视化
import numpy as np  # 导入 Numpy 库，用于数据处理
import openpyxl
import pandas as pd  # 导入 Pandas 库，用于数据处理
import torch  # 导入 PyTorch 库
import torch.nn as nn  # 导入 PyTorch 库中的神经网络模块，用于定义损失函数等
from torch.utils.data import Dataset, DataLoader  # 从 PyTorch 库中导入 Dataset 和 DataLoader 类，用于构建自定义数据集和数据加载器
from model import LstmNet  # 导入自定义的模型类 LstmNet
from torch.optim.lr_scheduler import CosineAnnealingLR

# 日志记录设置
logging = TNLog(dir='log', name='date at ' + time.strftime('%Y%m%d', time.localtime()))
# 定义参数
window_input_size = 24 * 4  # 输入窗口大小，即用多少小时的数据来预测未来
window_label_size = 4  # 输出窗口大小，即预测未来多少个小时的数据
epoch = 1000  # 训练轮数
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 设置设备类型，如果有 GPU 则使用 GPU，否则使用 CPU
batch_size = 16  # 批量大小,一次训练的样本数

input_size = 9  # 模型输入维度
hidden_size = 128  # 隐层维度
output_size = 1  # 模型输出维度


# 定义自定义数据集类
class MyDataset(Dataset):
    def __init__(self, data, window_input_size, window_label_size):
        self.data = pd.DataFrame(data)  # 将 data 转换为 Pandas 数据框
        self.window_input_size = window_input_size  # 输入窗口大小
        self.window_label_size = window_label_size  # 输出窗口大小
        self.window_gap = 8  # 滑动步长

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
    filename = "PV_train_time.csv"  # 数据文件名
    try:
        logging.info("读入数据")
        filepath = os.path.join(filedir, filename)  # 构造数据文件完整路径
        data = pd.read_csv(filepath, encoding="gbk")  # 读取 CSV 格式的数据文件
    except Exception as e:
        logging.error("读取数据失败, 失败原因为")
        logging.error(traceback.format_exc())
        raise e
    logging.info("数据读取成功")
    try:
        logging.info("时间列转换")
        data['时间'] = pd.to_datetime(data['时间'])
    except Exception as e:
        logging.error("时间列不存在")
        logging.error(traceback.format_exc())
        raise e
    logging.info("时间列转换成功")
    # 拆分时间列为年、月、日、小时、分钟四列
    data['year'] = data['时间'].dt.year
    data['month'] = data['时间'].dt.month
    data['day'] = data['时间'].dt.day
    data['hour'] = data['时间'].dt.hour
    # 删除原来的时间列
    data = data.drop(['时间'], axis=1)
    try:
        # 调整列的顺序，将新列插入到合适的位置
        new_order = ['year', 'month', 'day', 'hour', '功率', '光照强度', '温度', 'time_cos', 'time_sin', '湿度']  # 将"..."替换为原来的列名及其顺序
        data = data[new_order]
    except Exception as e:
        logging.error("数据修改失败")
        logging.error(traceback.format_exc())
        raise e
    try:
        logging.info("存储新的数据集")
        # 将处理后的数据写入到新的的CSV文件中, 不写入索引
        data.to_csv('handled_data/PV_train_time1.csv', index=False)
    except Exception as e:
        logging.error("数据存储失败,失败原因为")
        logging.error(traceback.format_exc())
        raise e
    logging.info("新数据集存储成功")
    # 划分训练集和验证集

    train_data = data[: int(len(data) * 0.78)]  # 前 80% 的数据作为训练集
    valid_data = data[int(len(data) * 0.78):]  # 后面 20% 的数据作为验证集

    try:
        logging.info("构建训练数据集")
        # 创建训练集和验证集的数据集和数据加载器
        Lstm_train_dataset = MyDataset(train_data, window_input_size, window_label_size)  # 构造训练数据集
        Lstm_valid_dataset = MyDataset(valid_data, window_input_size, window_label_size)  # 构造验证数据集
        Lstm_train_dataloader = DataLoader(Lstm_train_dataset, batch_size, shuffle=False)  # 构造训练数据加载器，不进行随机打乱操作
        Lstm_valid_dataloader = DataLoader(Lstm_valid_dataset, batch_size, shuffle=False)  # 构造验证数据加载器
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
val_losses = []  # 验证损失列表

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

    # 记录并打印验证损失
    Lstm_Net.eval()  # 开始验证，设评置为估模式
    batch_num_to_plot = 0
    with torch.no_grad():  # 不计算梯度，加速推理过程
        loss = []  # 损失列表
        output = 0
        for batch_num, (x, label) in enumerate(Lstm_valid_dataloader):  # 遍历验证数据加载器
            # 保存第一个batch的数据用于可视化
            x = x.clone().detach().to(device)  # 将输入放到指定设备上
            label = label.clone().detach().to(device)  # 将标签放到指定设备上
            output = Lstm_Net(x)  # 模型前向传递
            output = output[:, -window_label_size:, :]  # 取输出序列中最后的 window_label_size 个时间步
            predictions = output.cpu().numpy()  # 保存预测数据
            labels = label.cpu().numpy()  # 保存真实标签
            # 计算验证损失
            loss.append(criterion(output, label).item())  # 计算损失并将其存储到损失列表中

        val_losses.append(np.mean(loss))  # 计算验证损失的均值并存储到验证损失列表中
        print("Validation Epoch: %d, Loss: %.4f" % (i, np.mean(loss)))  # 打印验证损失

# 绘制真实数据与预测数据的对比曲线
plt.figure()
# 生成x轴刻度和标签
# x_ticks = np.arange(0, window_label_size, 1)
# x_tick_labels = ['7', '8', '9', '10']
try:
    logging.info("绘制图形")
    plt.plot(labels[0].flatten(), label='True Data')
    plt.plot(predictions[0].flatten(), label='Predictions')
except Exception as e:
    logging.error("绘制失败，失败原因为")
    logging.error(traceback.format_exc())
    raise e
logging.info("绘制成功")
# plt.xticks(x_ticks, x_tick_labels)
plt.xlabel('Time(h)')
plt.ylabel('Power(KW)')
plt.legend()
plt.savefig("output_results/data_comparison.png")  # 保存图像
plt.show()
plt.close()  # 关闭当前图形窗口

df_predictions = pd.DataFrame(predictions.flatten())

# 绘制损失曲线
plt.figure()
plt.plot(range(1, epoch + 1), train_losses, label="Training Loss")
plt.plot(range(1, epoch + 1), val_losses, label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

try:
    logging.info("保存模型参数")
    # 将模型的参数保存到文件中，文件名为 model.pt
    torch.save(Lstm_Net.state_dict(), "output_results/model.pt")
except Exception as e:
    logging.error("保存失败，失败原因为")
    logging.error(traceback.format_exc())
    raise e
logging.info("模型保存成功")
try:
    logging.info("保存预测结果")
    # 保存到 Excel 文件
    excel_filename = 'output_results/predictions.xlsx'
    workbook = openpyxl.Workbook()
    workbook.save(excel_filename)
    with pd.ExcelWriter(excel_filename, mode='a', engine='openpyxl', if_sheet_exists='overlay') as writer:
        df_predictions.to_excel(writer, index=False, sheet_name='Sheet1')
except Exception as e:
    logging.error("保存失败，失败原因为")
    logging.error(traceback.format_exc())
    raise e
logging.info("结果保存成功")
