import datetime
import numpy as np
import pandas as pd
from config import *
import psycopg2


def GetAllDataFromDB(row_name, table_name, tag_id, time='create_time', start_time=None):
    """
    :param row_name: 要选取的列名
    :param table_name: 表名
    :param tag_id: 用于在表中筛选需要查询数据的target_id
    :param time: 表中时间列的定义名称，默认为“create_time”
    :param start_time: 指定的起始时间，默认为空
    :return: 选取的数据
    """
    mydb = psycopg2.connect(
        host=dataaddress,  # 数据库主机地址
        port=port_num,
        user=user_name,  # 数据库用户名
        password=password,  # 数据库密码
        database=datause
    )
    cursor = mydb.cursor()
    if start_time is not None:
        sql = f"SELECT {row_name} FROM {table_name} WHERE tag_id='{tag_id}' AND {time} >= '{start_time}' ORDER BY {time} DESC"
    else:
        sql = f"SELECT {row_name} FROM {table_name} WHERE tag_id='{tag_id}' ORDER BY {time} DESC"
    cursor.execute(sql)
    data = pd.DataFrame(np.array(cursor.fetchall()))
    # 关闭数据库
    mydb.close()

    return data


def InsertData(table_name, data_dict):
    """
    :param table_name: 插入的表名称
    :param data_dict: 插入表中的数据，字典类型，key是数据库列名,value是相应的列对应的值。
    :return:
    """
    mydb = psycopg2.connect(
        host=dataaddress,  # 数据库主机地址
        port=port_num,
        user=user_name,  # 数据库用户名
        password=password,  # 数据库密码
        database=datause
    )
    cursor = mydb.cursor()

    # 写入时间生成
    write_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # 生成列名
    column_name = 'create_time'
    data = [write_time]
    for key in data_dict.keys():
        column_name = column_name + ',' + str(key)
        data.append(data_dict[key])
    data = tuple(data)

    # 生成sql语句
    tmp_s = '%' + 's'
    for key in data_dict.keys():
        tmp_s = tmp_s + ",%" + "s"  # 批量添加%s
    sql = """INSERT INTO public.""" + table_name + """(""" + column_name + """)
         VALUES (""" + tmp_s + """);"""  # 拼接sql语句

    # 执行sql语句
    cursor.execute(sql, data)
    mydb.commit()
    mydb.close()


def UpdateData(table_name, data_dict, tag_id):
    """
    :param table_name: 插入的表名称
    :param data_dict: 表中需要更新的数据，字典类型，key是数据库列名,value是相应的列对应的值。
    :param tag_id: 用于在表中筛选需要更新数据的target_id
    :return:
    """
    mydb = psycopg2.connect(
        host=dataaddress,  # 数据库主机地址
        port=port_num,
        user=user_name,  # 数据库用户名
        password=password,  # 数据库密码
        database=datause
    )
    cursor = mydb.cursor()

    # 写入时间生成
    write_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # 生成列名
    column_names = ['create_time']
    data = [write_time]
    for key in data_dict.keys():
        column_names.append(str(key))
        data.append(data_dict[key])

    # 生成sql语句
    tmp_s = '%' + 's'
    for key in data_dict.keys():
        tmp_s = tmp_s + ",%" + "s"  # 批量添加%s

    # 获取最新数据的时间戳
    latest_time_sql = f"SELECT MAX(create_time) FROM {table_name} WHERE tag_id = '{tag_id}'"
    cursor.execute(latest_time_sql)
    latest_time = cursor.fetchone()[0]

    set_statements = ", ".join([column_name + " = %s" for column_name in column_names])
    sql = """UPDATE public.""" + table_name + """
             SET """ + set_statements + """WHERE create_time = %s AND tag_id = %s;"""

    # 执行sql语句
    cursor.execute(sql, data + [latest_time, tag_id])
    mydb.commit()
    mydb.close()


if __name__ == '__main__':
    filepath = './output_results/predictions.xlsx'
    row_name = ['temperature']

    # 把predictions的数据导入到数据库里面
    df = pd.read_excel(filepath, sheet_name='Sheet1')
    for index, row in df.iterrows():
        # 将DataFrame数据转换为字典
        data_dict = row.to_dict()
        # 生成列名和对应的值，用于拼接SQL语句
        column_names = ','.join(data_dict.keys())
        values = [data_dict[key] for key in data_dict]
        tmp_s = ', '.join(['%s' for _ in range(len(values))])
        print(data_dict)
        InsertData(table_name=test_sheet, data_dict=data_dict)
        UpdateData(table_name=test_sheet, data_dict=data_dict, tag_id=area_id_3)
    # data_update_dict = {'temperature': 26.5, 'humidity': 24.3}
    # UpdateData(table_name=test_sheet, data_dict=data_update_dict, tag_id=area_id_3)

    data_all = np.array(GetAllDataFromDB(row_name=','.join(row_name), table_name=test_sheet, time='create_time', tag_id=area_id_1))
    print(data_all)

