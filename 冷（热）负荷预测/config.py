from log_print import TNLog
import time
import torch  # 导入 PyTorch 库
"""参数设置"""

# 日志记录设置
logging = TNLog(dir='log', name='date at ' + time.strftime('%Y%m%d', time.localtime()))
# 定义参数
window_input_size = 24 * 6  # 输入窗口大小，即用多少小时的数据来预测未来
window_label_size = 4  # 输出窗口大小，即预测未来多少个小时的数据
window_gap_size = 12
epoch = 12000  # 训练轮数
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 设置设备类型，如果有 GPU 则使用 GPU，否则使用 CPU
batch_size = 16  # 批量大小,一次训练的样本数

input_size = 9  # 模型输入维度
hidden_size = 128  # 隐层维度
output_size = 1  # 模型输出维度

# ems 数据库
"""数据库信息"""
dataaddress="123.249.70.226"  # 数据库主机地址
port_num="7047"  # 数据库端口号
user_name="root"  # 数据库用户名
password="ems123@4"  # 数据库密码
datause="ems"

"""数据库对接表"""
hvac_sheet = "ems_hvac_history"
els_sheet = "ems_els_history"
hydrogen_sheet = "ems_hydrogen_history"
output_sheet = "ems_hvac_load_forecast"

"""关口信息"""
area_id_1 = ""
area_id_2 = ""
area_id_3 = ""

# # 阿里云数据库
# """数据库信息"""
# dataaddress = "192.168.3.13"
# port_num = "3306"
# user_name = "root"
# password = "123456"
# datause = "ems"
