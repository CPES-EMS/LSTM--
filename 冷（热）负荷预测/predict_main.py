import datetime
import sys

import traceback
from LSTM_validation import Lstm_Net
from config import window_label_size, logging
from sql_lstm import GetAllDataFromDB, InsertData, UpdateData, GetNearestDataFromDB, IsExist
import pandas as pd
import torch
import numpy as np

Lstm_Net.load_state_dict(torch.load("./output_results/model.pt"))  # 加载模型参数

try:
    # 从电气系统数据库读入数据
    row_name1 = ['load_time', 'load_value']
    raw_data1 = GetAllDataFromDB(position_name='a', position_id='1', device_name='b', device_id='2',
                                 system_name='c',
                                 system_id='3', row_name=','.join(row_name1), table_name='sheet1', time='load_time',
                                 start_time=None)

    arr = np.array(raw_data1['load_time'])
    # 将ndarray转换为DataFrame
    df = pd.DataFrame(arr, columns=['datetime'])
    # 将datetime列转换为Datetime对象
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['year'] = df['datetime'].dt.year
    df['month'] = df['datetime'].dt.month
    df['day'] = df['datetime'].dt.day
    df['hour'] = df['datetime'].dt.hour
    week_list = []
    # 增添时间列为星期几
    for row in range(df.shape[0]):
        weekdays = df['datetime'][row]
        week_list.append(weekdays.weekday())
    df['weekdays'] = pd.DataFrame(week_list)
    # 对周末进行标记，用电量可能增多
    df.loc[:, 'weekend'] = 0
    df.loc[:, 'weekend_sat'] = 0
    df.loc[:, 'weekend_sun'] = 0
    df.loc[(df['weekdays'] > 4), 'weekend'] = 1
    df.loc[(df['weekdays'] == 5), 'weekend_sat'] = 1
    df.loc[(df['weekdays'] == 6), 'weekend_sun'] = 1
    df['load_value'] = raw_data1['load_value'][::-1].astype(float)

    # 夏天冷负荷，冬天热负荷，对夏天冬天进行标记
    month_list = []
    for row in range(df.shape[0]):
        months = df['month'][row]
        print(type(month_list))
        month_list.append(months)
    df['win_or_sum'] = pd.DataFrame(month_list)
    print(type(df['win_or_sum'].values))
    df.loc[:, 'summer'] = 0
    df.loc[:, 'winter'] = 0
    df.loc[(6 <= df['win_or_sum']), 'summer'] = 1  # 6月-9月定义为夏天
    df.loc[(df['win_or_sum'] >= 10), 'summer'] = 0
    df.loc[(df['win_or_sum'] >= 11), 'winter'] = 1  # 11月到次年3月定义为冬天
    df.loc[(df['win_or_sum'] <= 3), 'winter'] = 1

    raw_data1 = df
except Exception as e:
    logging.error("读取失败，失败原因为")
    logging.error(traceback.format_exc())
    raise e
logging.info("读取成功")

# 从其他表读入新数据
same_time = raw_data1['datetime']
raw_name2 = ['heat_value']
raw_data2 = GetNearestDataFromDB(position_name='a', position_id='1', device_name='b', device_id='2',
                                 system_name='c', target_time=same_time,
                                 system_id='3', row_name=','.join(raw_name2), table_name='sheet2', time='load_time',
                                 start_time=None)
df['heat_value'] = raw_data2['heat_value'].values[::-1].astype(float)
raw_data = df

# 对读取数据进行修改
time_before = raw_data['datetime']
new_order = ['year', 'month', 'day', 'hour', 'heat_value', 'load_value', 'weekend', 'weekend_sat',
                     'winter', 'weekend_sun','summer']
raw_data = raw_data[new_order]
raw_data = raw_data.iloc[0:24, 1:]
raw_data = raw_data[::-1]
print(raw_data)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_tensor = torch.tensor(raw_data.values)  # 转换为张量
data_tensor = data_tensor.unsqueeze(0).float().clone().detach().to(device)  # 增加一个维度以匹配模型的输入形状
output = Lstm_Net(data_tensor)  # 根据模型预测结果
# output = output[-window_label_size:, :]  # 取输出序列中最后的 window_label_size 个时间步
predictions = output.detach().cpu().numpy()  # 保存预测数据

predict_time = [(time_before + datetime.timedelta(hours=24))]
predictions = [inner_list[0] for outer_list in predictions for inner_list in outer_list]
for i in range(len(predictions)):
    predictions[i] = predictions[i].astype(float)
print(predictions)
# 将预测结果放入数据库
try:
    logging.info("插入数据")
    data_list = []
    for i in range(24):
        data_dict = {'system_id': '3', 'system_name': 'c', 'actual_time': predict_time[0][i],
                     'forcast_value': predictions[24-i-1], 'forcast_type': '24'}
        data_list.append(data_dict)
    print(data_list)
    for data_dict in data_list:
        if IsExist(data_dict) > 0:
            UpdateData(table_name='sheet3', data_dict=data_dict, system_name='c', system_id='3',
                       actual_time=data_dict['actual_time'])
        else:
            InsertData(table_name='sheet3', data_dict=data_dict)
except Exception as e:
    logging.error("插入数据失败, 失败原因为")
    logging.error(traceback.format_exc())
    raise e
logging.info("数据插入成功")
