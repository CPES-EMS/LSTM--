import mysql.connector
import sys
import warnings
from datetime import timedelta
import traceback

from LSTM_validation import Lstm_Net
from config import window_label_size, logging, els_sheet, hydrogen_sheet, hvac_sheet, output_sheet, env_sheet, \
    dataaddress, port_num, user_name, password, datause
from sql_lstm import GetAllDataFromDB, InsertData, UpdateData, GetNearestDataFromDB, GetPredictDataFromDB
import pandas as pd
import torch
import numpy as np

Lstm_Net.load_state_dict(torch.load("./output_results/model_24h.pt"))  # 加载模型参数
mydb = mysql.connector.connect(
    host=dataaddress,  # 数据库主机地址
    port=port_num,
    user=user_name,  # 数据库用户名
    passwd=password,  # 数据库密码
    database=datause
)


def get_data1():
    state = 0
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
            if temp_data1.shape[0] > 0:
                if i == 0:
                    raw_data1['load_time'] = temp_data1['load_time']
                    raw_data1[i] = temp_data1['load_value']
                else:
                    raw_data1[i] = temp_data1['load_value']
            else:
                logging.error('部分光伏设备数据为空')
        if raw_data1.shape[0] > 0:
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
        else:
            logging.error('全部光伏设备数据为空')
            raw_data1 = pd.DataFrame()
            state = 100
    except Exception as e:
        logging.error("读取失败，失败原因为")
        logging.error(traceback.format_exc())
        raise e
    finally:
        mydb.close()
    logging.info("读取成功")
    return raw_data1, state


def get_data2(raw_data1):
    try:
        same_time = raw_data1['load_time']
        row_name1 = ['load_time', 'load_value']
        # 从环境系统数据库读入天气预报数据
        position_name_list1 = ['室外温度（单位：℃）', '室外湿度（单位：％）']
        device_name_list1 = ['天气预报-温度', '天气预报-湿度']
        for j in range(0, len(position_name_list1)):
            temp_data2 = GetNearestDataFromDB(position_name=position_name_list1[j],
                                              device_name=device_name_list1[j],
                                              system_name='室外环境', target_time=same_time,
                                              system_id='0501', row_name=','.join(row_name1),
                                              table_name=env_sheet,
                                              time='load_time',
                                              start_time=None)
            if temp_data2.shape[0] > 0:
                # 对读入数据展平并转化为Series对象
                load_time1 = temp_data2['load_time'].values.flatten()
                load_value1 = temp_data2['load_value'].values.flatten()
                # 对异常值进行处理
                processed_data = []  # 存储处理后的数据
                for i in range(len(load_value1)):
                    if load_value1[i] < -100:
                        processed_data.append(load_value1[i - 1])
                    elif load_value1[i] > 100:
                        processed_data.append(load_value1[i - 1])
                    else:
                        processed_data.append(load_value1[i])
                processed_data = np.array(processed_data)
                temp_data2 = pd.DataFrame()
                temp_data2['load_time'] = load_time1
                temp_data2[j] = processed_data
                # 按照时间进行排序
                temp_data2 = temp_data2.groupby('load_time', as_index=False, sort=True).mean()
                print(temp_data2)
                if temp_data2.shape[0] != raw_data1.shape[0]:
                    warnings.warn("数据样本不统一")
                raw_data1 = pd.merge(raw_data1, temp_data2, how='outer', on='load_time', sort=True)
            else:
                logging.error('部分环境设备数据为空')
        raw_data1 = raw_data1.fillna(0)
    except Exception as e:
        logging.error("读取失败，失败原因为")
        logging.error(traceback.format_exc())
        raise e
    finally:
        mydb.close()
    logging.info("读取成功")
    return raw_data1

    # # 降采样处理得到间隔为1小时的数据
    # raw_data1['load_time'] = pd.to_datetime(raw_data1['load_time'])
    # raw_data1.set_index('load_time', inplace=True)
    # # 降采样为1小时数据，保留每小时内的第一个值
    # raw_data1 = raw_data1.resample('1H').first().shift(1, freq='10min')
    # raw_data1 = raw_data1.reset_index(drop=False)


def data_process(raw_data1):
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
        df['load_value'] = raw_data1['load_value'].astype(float)

        df['temperature'] = raw_data1[0].astype(float)
        df['humidity'] = raw_data1[1].astype(float)
        raw_data = df
        # print(raw_data)
        # 对读取数据进行修改
        if raw_data.shape[0] <= 24 * 12:
            raw_data = raw_data
        else:
            raw_data = raw_data.iloc[-24 * 12:, :]

        time_before = raw_data['datetime']
        time_before = time_before.reset_index(drop=True)
        new_order = ['year', 'month', 'day', 'hour', 'load_value', 'temperature',
                     'humidity']  # 将"..."替换为原来的列名及其顺序
        raw_data = raw_data[new_order]
        raw_data = raw_data.iloc[:, 1:]
        print(raw_data)
    except Exception as e:
        logging.error("读取失败，失败原因为")
        logging.error(traceback.format_exc())
        raise e
    finally:
        mydb.close()
    logging.info("读取成功")
    return raw_data, time_before


def predict_main():
    data, state = get_data1()
    if state == 0:
        data = get_data2(data)
        raw_data, time_before = data_process(data)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        data_tensor = torch.tensor(raw_data.values)  # 转换为张量
        data_tensor = data_tensor.unsqueeze(0).float().clone().detach().to(device)  # 增加一个维度以匹配模型的输入形状
        output = Lstm_Net(data_tensor)  # 根据模型预测结果
        output = output[-window_label_size:, :]  # 取输出序列中最后的 window_label_size 个时间步
        predictions = output.detach().cpu().numpy()  # 保存预测数据

        predict_time = [(time_before + timedelta(hours=24))]
        predictions = [inner_list[0] for outer_list in predictions for inner_list in outer_list]
        for i in range(len(predictions)):
            predictions[i] = predictions[i].astype(float)
            if predictions[i] < 0:
                predictions[i] = 0

        # print(predictions)
        # 将预测结果放入数据库

        try:
            logging.info("插入数据")
            for i in range(len(predictions)):
                data_dict = {'system_id': '10', 'system_name': '光伏功率',
                             'actual_time': predict_time[0][i].strftime('%Y-%m-%d %H:%M:%S'),
                             'forecast_value': float(predictions[i]), 'forecast_type': '24'}
                print(data_dict)
                # 读取现有预测数据
                predict_data_ori = np.array(
                    GetPredictDataFromDB(row_name='actual_time', table_name=output_sheet, actual_time='actual_time',
                                         system_id='10', system_name='光伏功率',
                                         forecast_type=data_dict['forecast_type']),
                    dtype='datetime64[D]')

                cur = np.datetime64(data_dict['actual_time']).astype('datetime64[D]')
                # print(type(cur))
                if np.isin(cur, predict_data_ori):
                    dic = dict()
                    dic['forecast_value'] = data_dict['forecast_value']
                    UpdateData(table_name=output_sheet, data_dict=dic, system_name='光伏功率', system_id='10',
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
        return 0
    else:
        logging.error('数据库为空，无法进行预测')
        return state


if __name__ == "__main__":
    state_code = predict_main()
