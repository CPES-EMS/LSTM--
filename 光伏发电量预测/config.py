from log_print import TNLog
import time
import torch  # 导入 PyTorch 库
"""参数设置"""

# 日志记录设置
logging = TNLog(dir='log', name='date at ' + time.strftime('%Y%m%d', time.localtime()))
# 定义参数
window_input_size = 24 * 4  # 输入窗口大小，即用多少小时的数据来预测未来
window_label_size = 4  # 输出窗口大小，即预测未来多少个小时的数据
epoch = 1000  # 训练轮数
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 设置设备类型，如果有 GPU 则使用 GPU，否则使用 CPU
batch_size = 16  # 批量大小,一次训练的样本数

input_size = 7  # 模型输入维度
hidden_size = 128  # 隐层维度
output_size = 1  # 模型输出维度

"""数据库信息"""
dataaddress="rm-2ze568o4qd438xw6cdo.mysql.rds.aliyuncs.com"  # 数据库主机地址
port_num="3306"  # 数据库端口号
user_name="yuce"  # 数据库用户名
password="yuce@123456"  # 数据库密码
datause="yuce"

"""数据库对接表"""
test_sheet = "ems_els_new_energy_electricity_forecast"
output_sheet = "ems_els_new_energy_electricity_forecast"

"""关口信息"""
area_id_1 = "DBC0441E15FDJ_310"
area_id_2 = "EAB020A_05B___302"
area_id_3 = "EAA050A222B___302"
