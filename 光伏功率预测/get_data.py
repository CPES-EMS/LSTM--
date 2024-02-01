from sql_lstm import GetAllDataFromDB, InsertData, UpdateData, GetNearestDataFromDB, GetPredictDataFromDB
from config import window_label_size, logging, els_sheet, hydrogen_sheet, hvac_sheet, output_sheet, dataaddress, \
    port_num, user_name, password, datause, device, env_sheet

import mysql.connector
import traceback
import pandas as pd
import numpy as np

mydb = mysql.connector.connect(
    host=dataaddress,  # 数据库主机地址
    port=port_num,
    user=user_name,  # 数据库用户名
    passwd=password,  # 数据库密码
    database=datause
)


def get_data1():
    raw_data1 = pd.DataFrame()
    state = 0
    try:
        # 从电气系统数据库读入光伏数据
        row_name1 = ['load_time', 'load_value']
        device_id_list = ['01010101', '01010102', '01010103', '01010104', '01010105', '01010106']
        position_id_list = ['01010101016388', '01010102016394', '01010103016400', '01010104016406',
                            '01010105016412', '01010106016418']
        for i in range(0, len(device_id_list)):
            temp_data1 = GetAllDataFromDB(position_id=position_id_list[i],
                                          device_id=device_id_list[i],
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
    except Exception:
        logging.error("读取失败，失败原因为")
        logging.error(traceback.format_exc())
        state = 106
    finally:
        mydb.close()
    logging.info("读取成功")
    return raw_data1, state


def get_data2(raw_data1):
    state = 0
    try:
        same_time = raw_data1['load_time']
        row_name1 = ['load_time', 'load_value']
        # 从环境系统数据库读入天气预报数据
        position_id_list1 = ['05012304002027',  '05012305002028']
        device_id_list1 = ['05012304', '05012305']
        for j in range(0, len(position_id_list1)):
            temp_data2 = GetNearestDataFromDB(position_id=position_id_list1[j],
                                              device_id=device_id_list1[j],
                                              target_time=same_time,
                                              system_id='0501', row_name=','.join(row_name1),
                                              table_name=env_sheet,
                                              time='load_time',
                                              start_time=None)
            print(temp_data2)
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

                if temp_data2.shape[0] != raw_data1.shape[0]:
                    logging.error("数据样本不统一")
                raw_data1 = pd.merge(raw_data1, temp_data2, how='outer', on='load_time', sort=True)
            else:
                logging.error('部分环境设备数据为空')
        raw_data1 = raw_data1.fillna(0)
    except Exception:
        logging.error("读取失败，失败原因为")
        logging.error(traceback.format_exc())
        state = 106
    finally:
        mydb.close()
    logging.info("读取成功")
    return raw_data1, state

