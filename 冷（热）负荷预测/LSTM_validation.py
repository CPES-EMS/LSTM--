import traceback
import numpy as np  # 导入 Numpy 库，用于数据处理
import openpyxl
import pandas as pd
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import DataLoader
from config import input_size, hidden_size, output_size, window_label_size, device, logging, window_input_size, \
    batch_size
from model import LstmNet
import torch
from LSTM_train import MyDataset

Lstm_Net = LstmNet(input_size, hidden_size, output_size, window_label_size).to(device)  # 构造 LSTM 模型，将
Lstm_Net.load_state_dict(torch.load("./output_results/model_24h.pt", map_location=device))  # 加载模型参数

try:
    logging.info("读入数据")
    # 预测数据
    filepath = './handled_data/yulin_load_elc.csv'
    data = pd.read_csv(filepath, encoding="utf-8")
except Exception as e:
    logging.error("读取数据失败, 失败原因为")
    logging.error(traceback.format_exc())
    raise e
logging.info("数据读取成功")
try:
    logging.info("构建验证数据集")
    test_data = data[int(len(data) * 0.80):]  # 选择最后 window_input_size 条数据作为测试数据
    test_dataset = MyDataset(test_data, window_input_size, window_label_size)  # 构造测试数据集
    test_dataloader = DataLoader(test_dataset, batch_size, shuffle=False)  # 构造测试数据加载器
except Exception as e:
    logging.error("构建失败，失败原因为")
    logging.error(traceback.format_exc())
    raise e
logging.info("构建成功")

# 使用smooth误差损失函数
criterion = nn.SmoothL1Loss()
val_losses = []  # 验证损失列表

Lstm_Net.eval()  # 开始验证，设评置为估模式
batch_num_to_plot = 0
with torch.no_grad():  # 不计算梯度，加速推理过程
    loss = []  # 损失列表
    output = 0
    for batch_num, (x, label) in enumerate(test_dataloader):  # 遍历验证数据加载器
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
#     print("Validation Loss: %.4f" % (np.mean(loss)))  # 打印验证损失
#
# # 绘制真实数据与预测数据的对比曲线
# plt.figure()
# # 生成x轴刻度和标签
# # x_ticks = np.arange(0, window_label_size, 1)
# # x_tick_labels = ['7', '8', '9', '10']
# try:
#     logging.info("绘制图形")
#     plt.plot(labels.flatten(), label='True Data')
#     plt.plot(predictions.flatten(), label='Predictions')
# except Exception as e:
#     logging.error("绘制失败，失败原因为")
#     logging.error(traceback.format_exc())
#     raise e
# logging.info("绘制成功")
# # plt.xticks(x_ticks, x_tick_labels)
# plt.xlabel('Time(h)')
# plt.ylabel('heat_load(KW)')
# plt.legend()
# plt.savefig("./output_results/data_comparison.png")  # 保存图像
# plt.show()
# plt.close()  # 关闭当前图形窗口

df_predictions = pd.DataFrame(predictions.flatten())

try:
    logging.info("保存预测结果")
    # 保存到 Excel 文件
    excel_filename = './output_results/predictions.xlsx'
    workbook = openpyxl.Workbook()
    workbook.save(excel_filename)
    with pd.ExcelWriter(excel_filename, mode='a', engine='openpyxl', if_sheet_exists='overlay') as writer:
        df_predictions.to_excel(writer, index=False, sheet_name='Sheet1', header=['predicted_power'])
except Exception as e:
    logging.error("保存失败，失败原因为")
    logging.error(traceback.format_exc())
    raise e

logging.info("结果保存成功")
