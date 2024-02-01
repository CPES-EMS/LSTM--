from sql_lstm import GetAllDataFromDB, InsertData, UpdateData, GetNearestDataFromDB, GetPredictDataFromDB
from config import window_label_size, logging, els_sheet, hydrogen_sheet, hvac_sheet, output_sheet, dataaddress, \
    port_num, user_name, password, datause, device

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


def get_data():

    state = 0
    try:
        # 从暖通系统数据库读入数据
        raw_data1 = pd.DataFrame()
        row_name1 = ['load_time', 'load_value']
        position_id_list = ['02011901000258', '02011901000262']
        for i in range(len(position_id_list)):
            temp_data1 = GetAllDataFromDB(position_id=position_id_list[i], device_id='02011901',
                                          system_id='0201', row_name=','.join(row_name1),
                                          table_name=hvac_sheet,
                                          time='load_time',
                                          start_time=None)
            if temp_data1.shape[0] > 0:
                if i == 0:
                    raw_data1['load_time'] = temp_data1['load_time']
                    raw_data1[i] = temp_data1['load_value']
                else:
                    raw_data1[i] = temp_data1['load_value']
            else:
                logging.error('部分暖通设备数据为空')
        if raw_data1.shape[0] > 0:
            # 求和得到总的热负荷
            raw_data1['load_value'] = raw_data1.iloc[:, 1:].sum(axis=1)
            # 对读入数据展平并转化为Series对象
            load_time = raw_data1['load_time'].values.flatten()
            load_value = raw_data1['load_value'].values.flatten()
            # 对异常值进行处理
            processed_data = []  # 存储处理后的数据
            for i in range(len(load_value)):
                if load_value[i] < -100:
                    processed_data.append(load_value[i - 1])
                elif load_value[i] > 100:
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
            raw_data1 = raw_data1.fillna(0)
            raw_data1['load_value'] = raw_data1['load_value'] * 100
            print(raw_data1)
        else:
            logging.error('全部暖通设备数据为空')
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


get_data()