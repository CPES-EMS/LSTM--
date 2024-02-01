import datetime
import time

import numpy as np
import pandas as pd

from config import *
import mysql.connector


def GetAllDataFromDB(position_id, device_id, system_id, row_name,
                     table_name, time='load_time', start_time=None):
    if len(position_id) == 0 or len(device_id) == 0 or len(system_id) == 0 \
            or row_name == False or table_name == False or time == False:
        raise ValueError("参数不能为空")
    """
    :param system_name:系统名称
    :param system_id:系统id
    :param position_id:采集点名称
    :param device_id: 设备名称
    :param row_name: 要选取的列名
    :param table_name: 表名
    :param time: 表中时间列的实际时间
    :param start_time: 指定的起始时间，默认为空
    :return: 选取的数据
    """
    mydb = mysql.connector.connect(
        host=dataaddress,  # 数据库主机地址
        port=port_num,
        user=user_name,  # 数据库用户名
        passwd=password,  # 数据库密码
        database=datause
    )
    cursor = mydb.cursor()

    # if start_time is not None: sql = f"SELECT {row_name} FROM {table_name} WHERE system_id ='{system_id}' AND {
    # time} >= '{start_time}' ORDER BY {time} DESC LIMIT 3" else:
    sql = f"SELECT {row_name} FROM {table_name} WHERE position_id='{position_id}' AND device_id='{device_id}' " \
          f"AND system_id ='{system_id}' ORDER BY {time} DESC "
    cursor.execute(sql)

    data = pd.DataFrame(cursor.fetchall(), columns=[row_name.split(",")])
    # 关闭数据库
    mydb.close()

    return data


def GetNearestDataFromDB(position_id, device_id, system_id, row_name,
                         table_name, target_time, time='load_time', start_time=None):
    if len(position_id) == 0 or len(device_id) == 0 or len(system_id) == 0 \
            or len(row_name) == 0 or len(table_name) == 0 or len(time) == 0 or len(target_time) == 0:
        raise ValueError("参数不能为空")
    """
    :param system_name:系统名称
    :param system_id:系统id
    :param position_id:采集点id
    :param device_id: 设备id
    :param row_name: 要选取的列名
    :param table_name: 表名
    :param time: 表中时间列的定义名称，默认为“create_time”
    :param target_time: 指定的目标时间
    :param start_time: 指定的起始时间，默认为空
    :return: 选取的数据
    """
    mydb = mysql.connector.connect(
        host=dataaddress,  # 数据库主机地址
        port=port_num,
        user=user_name,  # 数据库用户名
        passwd=password,  # 数据库密码
        database=datause
    )
    cursor = mydb.cursor()

    sql = f"SELECT {row_name} FROM {table_name} WHERE position_id='{position_id}' AND device_id='{device_id}' " \
          f"AND system_id ='{system_id}'" \
          f"ORDER BY ABS(TIMESTAMPDIFF(SECOND, {time}, '{target_time}')) ASC "
    cursor.execute(sql)
    data = pd.DataFrame(cursor.fetchall(), columns=[row_name.split(",")])
    # 关闭数据库
    mydb.close()

    return data


def InsertData(table_name, data_dict):
    if not all([table_name, data_dict]):
        raise ValueError("参数不能为空")
    """
    :param table_name: 插入的表名称
    :param data_dict: 插入表中的数据，字典类型，key是数据库列名,value是相应的列对应的值。
    :return:
    """
    mydb = mysql.connector.connect(
        host=dataaddress,  # 数据库主机地址
        port=port_num,
        user=user_name,  # 数据库用户名
        passwd=password,  # 数据库密码
        database=datause
    )
    cursor = mydb.cursor()

    # 写入时间生成
    write_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # 生成列名
    column_name = 'forecast_time'
    data = [write_time]
    for key in data_dict.keys():
        column_name = column_name + ',' + str(key)
        data.append(data_dict[key])
    data = tuple(data)

    # 生成sql语句
    tmp_s = '%' + 's'
    for key in data_dict.keys():
        tmp_s = tmp_s + ",%" + "s"  # 批量添加%s
    sql = """INSERT INTO """ + table_name + """(""" + column_name + """)
         VALUES (""" + tmp_s + """);"""  # 拼接sql语句

    # 执行sql语句
    cursor.execute(sql, data)
    mydb.commit()
    mydb.close()


def UpdateData(table_name, data_dict, area_name, area_id, actual_time):
    if len(area_name) == 0 or len(area_id) == 0 or data_dict == False \
            or table_name == False or actual_time == False:
        raise ValueError("参数不能为空")
    """
    :param actual_time:
    :param area_id:系统id
    :param area_name:系统名称
    :param table_name: 插入的表名称
    :param data_dict: 表中需要更新的数据，字典类型，key是数据库列名,value是相应的列对应的值。
    :return:
    """
    mydb = mysql.connector.connect(
        host=dataaddress,  # 数据库主机地址
        port=port_num,
        user=user_name,  # 数据库用户名
        passwd=password,  # 数据库密码
        database=datause
    )
    cursor = mydb.cursor()

    # 写入时间生成
    write_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # 生成列名
    column_names = ['forecast_time']
    data = [write_time]
    for key in data_dict.keys():
        column_names.append(str(key))
        data.append(data_dict[key])

    # 生成sql语句
    tmp_s = '%' + 's'
    for key in data_dict.keys():
        tmp_s = tmp_s + ",%" + "s"  # 批量添加%s

    # # 获取最新数据的时间戳
    # latest_time_sql = f"SELECT MAX(create_time) FROM {table_name} WHERE tag_id = '{tag_id}'"
    # cursor.execute(latest_time_sql)
    # latest_time = cursor.fetchone()[0]

    set_statements = ", ".join([column_name + " = %s" for column_name in column_names])
    sql = """UPDATE """ + table_name + """
             SET """ + set_statements + """ WHERE actual_time = %s AND area_name = %s AND area_id = %s;"""

    # 执行sql语句
    cursor.execute(sql, data + [actual_time, area_name, area_id])
    mydb.commit()
    mydb.close()


def GetPredictDataFromDB(row_name, table_name, actual_time, area_id, area_name, forecast_type):
    if len(area_name) == 0 or len(area_id) == 0 or len(forecast_type) == 0 \
            or row_name == False or table_name == False or actual_time == False:
        raise ValueError("参数不能为空")
    """
    :param forecast_type 预测类型
    :param area_name:地区名
    :param row_name:要选取的列名
    :param table_name:表名
    :param area_id:表中tag_id名称
    :param actual_time:表中时间列名称
    :return:
    """
    mydb = mysql.connector.connect(
        host=dataaddress,  # 数据库主机地址
        port=port_num,
        user=user_name,  # 数据库用户名
        passwd=password,  # 数据库密码
        database=datause
    )
    cursor = mydb.cursor()
    sql = f"SELECT {row_name} FROM {table_name} WHERE area_id='{area_id}' AND forecast_type='{forecast_type}' AND area_name='{area_name}' ORDER BY {actual_time} DESC"
    cursor.execute(sql)
    data = pd.DataFrame(np.array(cursor.fetchall()))
    mydb.close()
    return data


if __name__ == '__main__':

    predict_time = [(datetime.datetime.now() + datetime.timedelta(hours=i)).strftime('%Y-%m-%d %H:%M:%S')
                    for i in range(6)]
    data_list = [
        {'pre_sys_name': '光伏', 'region_id': "a", 'region_name': '榆林', 'system_name': "A",
         'actual_time': predict_time[0],
         'forcast_value': 1.23, 'forcast_type': 24},
        {'pre_sys_name': '光伏', 'region_id': "a", 'region_name': '榆林', 'system_name': "A",
         'actual_time': predict_time[1],
         'forcast_value': 2.23, 'forcast_type': 24},
        {'pre_sys_name': '光伏', 'region_id': "a", 'region_name': '榆林', 'system_name': "A",
         'actual_time': predict_time[2],
         'forcast_value': 3.23, 'forcast_type': 24},
    ]

    for data_dict in data_list:
        InsertData(table_name=output_sheet, data_dict=data_dict)

    replace_data_list = [
        {'actual_time': predict_time[3], 'forcast_value': 5.23, 'forcast_type': 24},
        {'actual_time': predict_time[4], 'forcast_value': 6.23, 'forcast_type': 24},
        {'actual_time': predict_time[5], 'forcast_value': 7.23, 'forcast_type': 24},
    ]

    row_name = ['actual_time', 'forcast_time', 'forcast_value', 'forcast_type']
    data_all = GetAllDataFromDB(pre_sys_name='光伏', region_name='榆林', row_name=','.join(row_name),
                                table_name=output_sheet, region_id='a', forcast_type=24)
    data_all = data_all[::-1]
    print(data_all)

    for index, row in data_all.iterrows():
        actual_time = str(row['actual_time'])
        region_id = str(row['region_id'])
        pre_sys_name = str(row['pre_sys_name'])
        UpdateData(table_name=els_sheet, data_dict=replace_data_list[index], pre_sys_name='光伏', region_name='榆林',
                   region_id=region_id, actual_time=actual_time)
