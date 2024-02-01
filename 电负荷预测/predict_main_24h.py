import sys

import mysql.connector
import datetime
from datetime import timedelta
import traceback
from get_data import *
from config import window_label_size, logging, els_sheet, hydrogen_sheet, hvac_sheet, output_sheet, dataaddress, \
    port_num, user_name, password, datause, device, input_size, hidden_size, output_size
from model import LstmNet
from sql_lstm import GetAllDataFromDB, InsertData, UpdateData, GetNearestDataFromDB, GetPredictDataFromDB
import pandas as pd
import torch
import numpy as np

Lstm_Net = LstmNet(input_size, hidden_size, output_size, window_label_size).to(device)  # 构造 LSTM 模型，将
Lstm_Net.load_state_dict(torch.load("./output_results/model_24h.pt", map_location=device))  # 加载模型参数
mydb = mysql.connector.connect(
    host=dataaddress,  # 数据库主机地址
    port=port_num,
    user=user_name,  # 数据库用户名
    passwd=password,  # 数据库密码
    database=datause
)


try:
    def data_process(raw_data1):

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
        # print(raw_data)

        # 对读取数据进行修改
        if raw_data.shape[0] <= 24 * 12:
            raw_data = raw_data
        else:
            raw_data = raw_data.iloc[-24 * 12:, :]

        time_before = raw_data['datetime']
        time_before = time_before.reset_index(drop=True)
        new_order = ['year', 'month', 'day', 'hour', 'load_value', 'weekend', 'weekend_sat',
                     'weekend_sun']  # 将"..."替换为原来的列名及其顺序
        raw_data = raw_data[new_order]
        raw_data = raw_data.iloc[:, 1:]
        return raw_data, time_before
except Exception:
    logging.error("添加时间列失败，失败原因为")
    logging.error(traceback.format_exc())
logging.info("添加成功")


def predict_main():
    data, state = get_data1()
    data, state2 = get_data2(data)
    data, state3 = get_data3(data)
    data, state4 = get_data4(data)
    if state == 0 and state2 == 0 and state3 == 0 and state4 == 0:

        raw_data, time_before = data_process(data)
        # 调用模型进行预测
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        data_tensor = torch.tensor(raw_data.values)  # 转换为张量
        data_tensor = data_tensor.unsqueeze(0).float().clone().detach().to(device)  # 增加一个维度以匹配模型的输入形状
        output = Lstm_Net(data_tensor)  # 根据模型预测结果
        # output = output[-window_label_size:, :]  # 取输出序列中最后的 window_label_size 个时间步
        predictions = output.detach().cpu().numpy()  # 保存预测数据

        predict_time = [(time_before + timedelta(hours=24))]
        # print(len(predict_time[0]))
        predictions = [inner_list[0] for outer_list in predictions for inner_list in outer_list]
        for i in range(len(predictions)):
            predictions[i] = predictions[i].astype(float)
            if predictions[i] < 0:
                predictions[i] = 0

        # 将预测结果放入数据库

        try:
            logging.info("插入数据")
            for i in range(len(predictions)):
                data_dict = {'area_id': '10', 'area_name': '电负荷',
                             'actual_time': predict_time[0][i].strftime('%Y-%m-%d %H:%M:%S'),
                             'forecast_value': float(predictions[i]), 'forecast_type': '24'}
                # print(data_dict)
                # 读取现有预测数据
                predict_data_ori = np.array(
                    GetPredictDataFromDB(row_name='actual_time', table_name=output_sheet, actual_time='actual_time',
                                         area_id='10', area_name='电负荷', forecast_type=data_dict['forecast_type']),
                    dtype='datetime64[s]')
                cur = np.datetime64(data_dict['actual_time']).astype('datetime64[s]')
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
            state = 107
        finally:
            mydb.close()
        logging.info("数据插入成功")
        return state
    else:
        logging.error('数据库为空或数据库无法读取，无法进行预测')
        return max(state, state2, state3, state4)


if __name__ == "__main__":
    state_code = predict_main()
