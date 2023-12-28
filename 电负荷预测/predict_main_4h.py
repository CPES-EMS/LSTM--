import sys

import mysql.connector
import datetime
import warnings
from datetime import timedelta
import traceback

from LSTM_validation import Lstm_Net
from config import window_label_size, logging, els_sheet, hydrogen_sheet, hvac_sheet, output_sheet, dataaddress, \
    port_num, user_name, password, datause
from sql_lstm import GetAllDataFromDB, InsertData, UpdateData, GetNearestDataFromDB, GetPredictDataFromDB
import pandas as pd
import torch
import numpy as np

Lstm_Net.load_state_dict(torch.load("./output_results/model_4h.pt"))  # 加载模型参数
mydb = mysql.connector.connect(
    host=dataaddress,  # 数据库主机地址
    port=port_num,
    user=user_name,  # 数据库用户名
    passwd=password,  # 数据库密码
    database=datause
)

try:
    # 从电气系统数据库读入光伏数据
    row_name1 = ['load_time', 'load_value']
    device_name_list = ['DC/DC-1号DC/DC', 'DC/DC-2号DC/DC', 'DC/DC-3号DC/DC', 'DC/DC-4号DC/DC',
                        'DC/DC-5号DC/DC', 'DC/DC-6号DC/DC']
    position_name_list = ['1#总功率', '2#总功率', '3#总功率', '4#总功率', '5#总功率', '6#总功率']
    raw_data1 = pd.DataFrame()
    for i in range(0, len(device_name_list)):
        temp_data1 = GetAllDataFromDB(position_name=position_name_list[i],
                                      device_name=device_name_list[i],
                                      system_name='光伏系统',
                                      system_id='0101', row_name=','.join(row_name1),
                                      table_name=els_sheet,
                                      time='load_time',
                                      start_time=None)
        if temp_data1.shape[0] == 0:
            logging.error("数据库的数据为空")
            sys.exit(100)

        if i == 0:
            raw_data1['load_time'] = temp_data1['load_time']
            raw_data1[i] = temp_data1['load_value']
        else:
            raw_data1[i] = temp_data1['load_value']

    # 求和得到总的光伏功率
    raw_data1['load_value'] = raw_data1.iloc[:, 1:].sum(axis=1)
    # 对读入数据展平并转化为Series对象
    load_time = raw_data1['load_time'].values.flatten()
    load_value = raw_data1['load_value'].values.flatten()
    # 对异常值进行处理
    processed_data = []  # 存储处理后的数据
    for i in range(len(load_value)):
        if load_value[i] < -100:
            processed_data.append(load_value[i - 1])
        elif load_value[i] > 100000:
            processed_data.append(load_value[i - 1])
        else:
            processed_data.append(load_value[i])
    processed_data = np.array(processed_data)
    raw_data1 = pd.DataFrame()
    raw_data1['load_time'] = load_time
    raw_data1['load_value'] = processed_data
    # 按时间分组并求均值
    raw_data1 = raw_data1.groupby('load_time', as_index=False, sort=True).mean()

    order = ['load_time', 'load_value']
    raw_data1 = raw_data1[order]
    raw_data1 = raw_data1.reset_index(drop=True)

except Exception as e:
    logging.error("读取失败，失败原因为")
    logging.error(traceback.format_exc())
    raise e
finally:
    mydb.close()
logging.info("读取成功")

try:

    # 从电气系统数据库读入买电量数据
    same_time = raw_data1['load_time']
    raw_name2 = ['load_time', 'load_value']
    position_name_list1 = ['1号隔离变开关柜有功总功率', '2号隔离变开关柜有功总功率', '3号隔离变开关柜有功总功率']
    device_name_list1 = ['能源站电表-1号能源站电表', '能源站电表-2号能源站电表', '能源站电表-3号能源站电表']
    for j in range(len(position_name_list1)):
        temp_data2 = GetNearestDataFromDB(position_name=position_name_list1[j],
                                          device_name=device_name_list1[j],
                                          system_name='电表', target_time=same_time,
                                          system_id='0106', row_name=','.join(raw_name2),
                                          table_name=els_sheet,
                                          time='load_time',
                                          start_time=None)

        if temp_data2.shape[0] == 0:
            logging.error("数据库的数据为空")
            sys.exit(100)
        # 对读入数据展平并转化为Series对象
        load_time1 = temp_data2['load_time'].values.flatten()
        load_value1 = temp_data2['load_value'].values.flatten()
        # 对异常值进行处理
        processed_data1 = []  # 存储处理后的数据
        for i in range(len(load_value1)):
            if load_value1[i] < -100:
                processed_data1.append(load_value1[i - 1])
            elif load_value1[i] > 100000:
                processed_data1.append(load_value1[i - 1])
            else:
                processed_data1.append(load_value1[i])
        processed_data1 = np.array(processed_data1)
        temp_data2 = pd.DataFrame()
        temp_data2['load_time'] = load_time1
        temp_data2[j] = processed_data1
        # 按时间分组并求均值
        temp_data2 = temp_data2.groupby('load_time', as_index=False, sort=True).mean()
        if temp_data2.shape[0] != raw_data1.shape[0]:
            warnings.warn("数据样本不统一")
        raw_data1 = pd.merge(raw_data1, temp_data2, how='outer', on='load_time', sort=True)
    raw_data1 = raw_data1.fillna(0)

except Exception as e:
    logging.error("读取失败，失败原因为")
    logging.error(traceback.format_exc())
    raise e
finally:
    mydb.close()
logging.info("读取成功")

try:
    # 从氢能系统数据库读入燃料电池数据
    temp_data3 = GetNearestDataFromDB(position_name='系统输出总功率', device_name='氢燃料电池-1号氢燃料电池',
                                      system_name='用氢系统', target_time=same_time,
                                      system_id='0303', row_name=','.join(row_name1),
                                      table_name=hydrogen_sheet,
                                      time='load_time',
                                      start_time=None)
    if temp_data3.shape[0] == 0:
        logging.error("数据库的数据为空")
        sys.exit(100)
    # 对读入数据展平并转化为Series对象
    load_time2 = temp_data3['load_time'].values.flatten()
    load_value2 = temp_data3['load_value'].values.flatten()
    # 对异常值进行处理
    processed_data2 = []  # 存储处理后的数据
    for i in range(len(load_value2)):
        if load_value2[i] < -100:
            processed_data2.append(load_value2[i - 1])
        elif load_value2[i] > 100000:
            processed_data2.append(load_value2[i - 1])
        else:
            processed_data2.append(load_value2[i])
    processed_data2 = np.array(processed_data2)
    temp_data3 = pd.DataFrame()
    temp_data3['load_time'] = load_time2
    temp_data3[3] = processed_data2
    # 按时间分组并求均值
    temp_data3 = temp_data3.groupby('load_time', as_index=False, sort=True).mean()
    # temp_data3 = temp_data3[:-5]
    if temp_data3.shape[0] != raw_data1.shape[0]:
        warnings.warn("数据样本不统一")
    raw_data1 = pd.merge(raw_data1, temp_data3, how='outer', on='load_time', sort=True)
    raw_data1 = raw_data1.fillna(0)

    # 求和得到总的用电
    raw_data1['load_value'] = raw_data1.iloc[:, 1:].sum(axis=1)
    order = ['load_time', 'load_value']
    raw_data1 = raw_data1[order]
    print(raw_data1)

except Exception as e:
    logging.error("读取失败，失败原因为")
    logging.error(traceback.format_exc())
    raise e
finally:
    mydb.close()
logging.info("读取成功")

try:
    # 从电气系统读入各个设备厂用电数据
    raw_name3 = ['load_time', 'load_value']
    position_name_list2 = ['1A7有功功率', '2A4有功功率', '2A5有功功率', '3A6有功功率', '1A4-1地源热泵有功功率']
    device_name_list2 = ['电锅炉厂用电-1A7', '电锅炉厂用电-2A4', '电锅炉厂用电-2A5', '电锅炉厂用电-3A6', '地源热泵厂用电-1A4-1地源热泵']
    for k in range(len(position_name_list2)):
        temp_data4 = GetNearestDataFromDB(position_name=position_name_list2[k],
                                          device_name=device_name_list2[k],
                                          system_name='厂用电', target_time=same_time,
                                          system_id='0107', row_name=','.join(raw_name3),
                                          table_name=els_sheet,
                                          time='load_time',
                                          start_time=None)

        if temp_data4.shape[0] == 0:
            logging.error("数据库的数据为空")
            sys.exit(100)
        # 对读入数据展平并转化为Series对象
        load_time3 = temp_data4['load_time'].values.flatten()
        load_value3 = temp_data4['load_value'].values.flatten()
        # 对异常值进行处理
        processed_data3 = []  # 存储处理后的数据
        for i in range(len(load_value3)):
            if load_value3[i] < -100:
                processed_data3.append(load_value3[i - 1])
            elif load_value3[i] > 100000:
                processed_data3.append(load_value3[i - 1])
            else:
                processed_data3.append(load_value3[i])
        processed_data3 = np.array(processed_data3)
        temp_data4 = pd.DataFrame()
        temp_data4['load_time'] = load_time3
        temp_data4[k] = processed_data3
        # 按时间分组并求均值
        temp_data4 = temp_data4.groupby('load_time', as_index=False, sort=True).mean()
        # if temp_data4.shape[0] != raw_data1.shape[0]:
        #     warnings.warn("数据样本不统一")
        raw_data1 = pd.merge(raw_data1, temp_data4, how='outer', on='load_time', sort=True)
    raw_data1 = raw_data1.fillna(0)
    print(raw_data1)

    # 求和得到总的耗电量
    raw_data1['use'] = raw_data1.iloc[:, 2:].sum(axis=1)
    # 用电量减去耗电量得到总的电负荷
    raw_data1['load_value'] = raw_data1['load_value'] - raw_data1['use']
    order = ['load_time', 'load_value']
    raw_data1 = raw_data1[order]
    print(raw_data1)

#     # 降采样处理得到间隔为1小时的数据
#     raw_data1['load_time'] = pd.to_datetime(raw_data1['load_time'])
#     raw_data1.set_index('load_time', inplace=True)
#     # 降采样为1小时数据，保留每小时内的第一个值
#     raw_data1 = raw_data1.resample('1H').first().shift(1, freq='10min')
#     raw_data1 = raw_data1.reset_index(drop=False)
except Exception as e:
    logging.error("读取失败，失败原因为")
    logging.error(traceback.format_exc())
    raise e
finally:
    mydb.close()
logging.info("读取成功")

try:

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
    df['load_value'] = raw_data1['load_value'].astype(float)
    raw_data = df
except Exception as e:
    logging.error("添加时间列失败，失败原因为")
    logging.error(traceback.format_exc())
    raise e
logging.info("添加成功")
print(raw_data)

# 对读取数据进行修改
if raw_data.shape[0] <= 4 * 12:
    raw_data = raw_data
else:
    raw_data = raw_data.iloc[-4 * 12:, :]

time_before = raw_data['datetime']
time_before = time_before.reset_index(drop=True)
new_order = ['year', 'month', 'day', 'hour', 'load_value', 'weekend', 'weekend_sat',
             'weekend_sun']  # 将"..."替换为原来的列名及其顺序
raw_data = raw_data[new_order]
raw_data = raw_data.iloc[:, 1:]

# 调用模型进行预测
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_tensor = torch.tensor(raw_data.values)  # 转换为张量
data_tensor = data_tensor.unsqueeze(0).float().clone().detach().to(device)  # 增加一个维度以匹配模型的输入形状
output = Lstm_Net(data_tensor)  # 根据模型预测结果
# output = output[-window_label_size:, :]  # 取输出序列中最后的 window_label_size 个时间步
predictions = output.detach().cpu().numpy()  # 保存预测数据

predict_time = [(time_before + timedelta(hours=4))]
# print(len(predict_time[0]))
predictions = [inner_list[0] for outer_list in predictions for inner_list in outer_list]
for i in range(len(predictions)):
    predictions[i] = predictions[i].astype(float)


# 将预测结果放入数据库

try:
    logging.info("插入数据")
    for i in range(len(predictions)):
        data_dict = {'area_id': '10', 'area_name': '电负荷', 'actual_time': predict_time[0][i].strftime('%Y-%m-%d %H:%M:%S'),
                     'forecast_value': float(predictions[i]), 'forecast_type': '4'}
        print(data_dict)
        # 读取现有预测数据
        predict_data_ori = np.array(
            GetPredictDataFromDB(row_name='actual_time', table_name=output_sheet, actual_time='actual_time',
                                 area_id='10', area_name='电负荷', forecast_type=data_dict['forecast_type']), dtype='datetime64[D]')
        cur = np.datetime64(data_dict['actual_time']).astype('datetime64[D]')
        if np.isin(cur, predict_data_ori):
            dic = dict()
            dic['forecast_value'] = data_dict['forecast_value']
            UpdateData(table_name=output_sheet, data_dict=dic, area_name='电负荷', area_id='10',
                       actual_time=data_dict['actual_time'])
        else:
            InsertData(table_name=output_sheet, data_dict=data_dict)
except Exception as e:
    logging.error("插入数据失败, 失败原因为")
    logging.error(traceback.format_exc())
    raise e
finally:
    mydb.close()
logging.info("数据插入成功")
