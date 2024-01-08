import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
import torch.nn.functional as F
import numpy as np
import collections
from PIL import Image
import csv
import random
import pandas as pd
import json


    
class xwdataset(Dataset):
    def __init__(self, root, mode = 'train', twindow = 20, batchsz = 5, n_way = 5, k_shot = 1, k_query = 15, startidx=0, valid_per = 0.4):
        ####################################################################################################################
        # 设定整个数据集的路径
        # 获取当前运行文件路径
        print('root:', root)
        self.path = root
        # 先定义一些全局变量，经过筛选后得到的连续量和状态量的参数名
        # 连续量
        self.num_vars = ['POWER1模块母线电流遥测', '+Y太阳阵电流遥测2(POWER1模块)', 'POWER2模块母线电流遥测', '+Y太阳阵电流遥测3(POWER2模块)',
                    'POWER3模块母线电流遥测', '-Y太阳阵电流遥测3(POWER2模块)', 'CDM1模块母线电流遥测', '-Y太阳阵电流遥测1(POWER3模块)',
                    'CDM2模块母线电流遥测', '-Y太阳阵电流遥测2(POWER3模块)', 'BCI模块母线电流遥测', '+Y太阳阵电流遥测1(POWER1模块)',
                    '+X隔板温度', '-Y板温度1', '-Y板温度2', '-Y板温度3',
                    '锂离子蓄电池A模块1温度', '-Y太阳翼外板温度', '-Y太阳翼中板温度', '-Y太阳翼内板温度',
                    '太阳翼SMA拉断器（-Y）1温度', '太阳翼SMA拉断器（-Y）2温度', '太阳翼SMA拉断器（-Y）3温度', '太阳翼SMA拉断器（-Y）4温度',
                    '锂离子蓄电池A模块2温度', '太阳翼SMA拉断器（-Y）5温度', '太阳翼SMA拉断器（-Y）6温度', '锂离子蓄电池B模块1温度',
                    'BCI模块温度遥测', 'POWER1模块温度遥测', 'POWER2模块温度遥测', 'CDM1模块温度遥测',
                    'POWER3模块温度遥测', 'CDM2模块温度遥测', '锂离子蓄电池B模块2温度', '锂离子蓄电池C模块1温度',
                    '锂离子蓄电池C模块2温度', '电源调节与配电单元温度', '母线电压遥测', 'TMTC备份+12V电压遥测',
                    'TMTC备份+5V电压遥测', 'TMTC主份+12V电压遥测', 'TMTC主份+5V电压遥测', '过充保护阈值回读',
                    '5V参考电压遥测', 'BCM电压遥测', '蓄电池电压遥测', '充电电压挡位遥测',
                    '第1路蓄电池远端电压采集', '第2路蓄电池远端电压采集', '充电电流挡位遥测']

        # 状态量
        self.state_vars = ['68W热控支路3通断状态', '-Y板加热器1通断状态', '+Y太阳翼6号压紧点主份拉断器通断状态',
                        '-Y太阳翼2号压紧点主份拉断器通断状态', '-Y太阳翼5号压紧点主份拉断器通断状态', '+Y板加热器3通断状态',
                        '+Y板加热器2通断状态', '68W热控支路7通断状态', '蓄电池加热器2通断状态',
                        '68W热控支路4通断状态', 'S3R22保护状态', 'S319保护状态',
                        'S3R16保护状态', 'S313保护状态', 'S3R10保护状态',
                        'S3R7保护状态', 'S3R4保护状态', 'S3R1保护状态',
                        'SATM(-Y)备份加热器通断状态', '安全开关J通断状态', '安全开关I通断状态',
                        '-Y太阳翼3号压紧点备份拉断器通断状态', '安全开关G通断状态', '-Y太阳翼4号压紧点备份拉断器通断状态',
                        '安全开关F通断状态', '+Y太阳翼3号压紧点备份拉断器通断状态', '+Y太阳翼2号压紧点备份拉断器通断状态',
                        '+Y太阳翼5号压紧点备份拉断器通断状态', '-Y太阳翼6号压紧点备份拉断器通断状态', '安全开关H通断状态',
                        '-Y太阳翼1号压紧点备份拉断器通断状态', '+Y太阳翼4号压紧点备份拉断器通断状态', '+Y太阳翼1号压紧点备份拉断器通断状态',
                        '68W热控支路10通断状态', 'u电池加热器3通断状态', '+Y太阳翼6号压紧点备份拉断器通断状态',
                        '-Y太阳翼2号压紧点备份拉断器通断状态', '-Y太阳翼5号压紧点备份拉断器通断状态', '+Y板加热器4通断状态',
                        '-Y板加热器5通断状态', '68W热控支路14通断状态', '-Y板加热器3通断状态',
                        '+Y板加热器6通断状态', '68W热控支路11通断状态', 'S3R24保护状态',
                        'S3R21保护状态', 'S3R18保护状态', 'S3R15保护状态',
                        'S3R12保护状态', 'S3R9保护状态', 'S3R6保护状态',
                        'S3R3保护状态']

        # 输入输出参数名保存为文件
        self.idx = max(len(self.state_vars), len(self.num_vars))
        df = pd.DataFrame()
        float_vars = pd.Series(self.num_vars + [np.nan] * (self.idx - len(self.num_vars))) # 用nan补齐
        int_vars = pd.Series(self.state_vars + [np.nan] * (self.idx - len(self.state_vars)))
        df['float'] = float_vars
        df['int'] = int_vars
        df.to_csv(os.path.join(self.path,'参数名.csv'), encoding='utf-8-sig', index=False)

        ####################################################################################################################
        # 将故障类型取值用字典保存，存储为json文件
        self.all_fault = ['电源控制器-BCRB模块-MEA电路中单个驱动三级管开路或短路', '电源控制器-BCRB模块-MEA电路中单个运放开路或短路',
                        '电源控制器-BCRB模块-蓄电池充电电压档位误指令', '电源控制器-电源下位机-CDM模块电流遥测电路故障',
                        '电源控制器-电源下位机-POWER模块母线电流测量电路故障', '电源控制器-电源下位机-太阳阵电流测量电路故障',
                        '电源控制器-电源下位机-通讯故障', '电源控制器-负载短路',
                        '电源控制器-功率模块-S3R电路二极管短路', '电源控制器-功率模块-S3R电路分流管MOSFET开路',
                        '电源控制器-功率模块-S3R分流状态异常', '电源控制器-配电（加热器）模块--Y板加热器误通',
                        '电源控制器-配电（加热器）模块-蓄电池加热带误断', '电源控制器-配电（加热器）模块-蓄电池加热带误通',
                        '太阳电池阵-隔离二极管短路', '太阳电池阵-隔离二极管开路',
                        '太阳电池阵-互连片开路', '太阳电池阵-汇流条焊点开路',
                        '太阳电池阵-太阳电池片-单片短路', '太阳电池阵-太阳电池片-单片开路',
                        '太阳电池阵-太阳电池片-太阳电池片性能衰降', '太阳电池阵-太阳翼-单分阵开路',
                        '太阳电池阵-太阳翼-单翼开路', '太阳电池阵-太阳翼-太阳翼单子阵开路']
        self.fault_dic = dict()
        for i in range(len(self.all_fault)):
            self.fault_dic[self.all_fault[i]] = i
        with open(os.path.join(self.path,'故障类型.json'), 'w', encoding='utf-8-sig') as file_json:
            json.dump(self.fault_dic, file_json, indent=4, ensure_ascii=False)

        ####################################################################################################################
        ## meta-learning的相关参数设定
        self.batchsz = batchsz  # batch of set, not batch of imgs
        self.twindow = twindow  # time window, 用于划分数据集
        self.n_way = n_way  # n-way
        self.k_shot = k_shot  # k-shot
        self.k_query = k_query  # for evaluation
        self.setsz = self.n_way * self.k_shot  # num of samples per set # 每个任务的支持集的样本数，等于n_way * k_shot
        self.querysz = self.n_way * self.k_query  # number of samples per set for evaluation # 每个任务的查询集的样本数，等于n_way * k_query
        self.startidx = startidx  # index label not from 0, but from startidx
        print('shuffle DB :%s, b:%d, %d-way, %d-shot, %d-query' % (mode, batchsz, n_way, k_shot, k_query))
        ####################################################################################################################
        # 类别数是self.label中不同元素的个数
        self.num_classes = len(self.all_fault) # set()函数创建一个无序不重复元素集
        
        # 载入数据，x1为模拟量，x2为状态量，y为标签
        self.x_num, self.x_state, self.y = self.load_data(self.path)

        # # 划分数据集
        # self.x_num_train, self.x_state_train, self.y_train, self.x_num_valid, self.x_state_valid, self.y_valid = self.__split_data__(self.x_num, self.x_state, self.y, per = 1 - valid_per)

        # # 数据标准化
        # data_mean, data_std, self.x_num_train, self.x_num_valid = self.__data_normalization__(self.x_num_train, self.x_num_valid)

        # # 数据标准化
        data_mean, data_std, self.x_num = self.__full_data_normalization__(self.x_num)

        # 保存均值和标准差
        mean_std = np.concatenate((data_mean.reshape([-1, 1]), data_std.reshape([-1, 1])), axis=1)
        mean_std = pd.DataFrame(mean_std, columns=['均值', '标准差'], index=self.num_vars)
        mean_std.to_csv(os.path.join(self.path,'变量均值标准差_诊断.csv'), encoding='utf-8-sig') 


        # 统计各个故障的样本量
        self.fault_num = np.zeros(self.num_classes)
        for i in range(len(self.y)):
            self.fault_num[np.argmax(self.y[i])] += 1
        print(self.fault_num)

        
        
    ## 载入数据集
    def load_data(self, path):
        data_origin = pd.read_csv(os.path.join(path,'fault_data.csv')) # 读取csv文件
        vars_all = pd.read_csv(os.path.join(self.path,'参数名.csv')) # 读取参数名文件
        num_vars = list(vars_all.iloc[:,0].dropna()) # 读取连续量参数名
        state_vars = list(vars_all.iloc[:,1].dropna()) # 读取状态量参数名
        
        data_label = data_origin['label'].copy() # 读取故障类型
        data_num = data_origin[num_vars].copy() # 读取连续量
        data_state = data_origin[state_vars].copy() # 读取状态量
        file_num = data_origin['No.'].copy() # 读取文件编号
        
        x1 = np.empty((0, self.twindow, data_num.shape[-1]), dtype=np.float32)
        x2 = np.empty((0, self.twindow, data_state.shape[-1]), dtype=np.int32)
        y = np.empty((0, self.num_classes), dtype=np.int32)
        
        for i in range(np.max(file_num+1)):
            position = (file_num == i) # 选出第i个文件的数据，每个文件对应了一个故障类型，即一个标签，所以每个文件的数据都是同一类故障
            branch_num = self.__add_window__(data_num[position], self.twindow, dtype=np.float32)  # 加时滞窗，升维度
            branch_state = self.__add_window__(data_state[position], self.twindow, dtype=np.int32)  # 加时滞窗，升维度
            plb = np.array(data_label[position], dtype=np.int32)[self.twindow - 1:] # 标签
            plb = self.__to_categorical__(plb, len(self.all_fault)) # 独热编码，这里其实是多分类问题，所以标签是独热编码，但其实在每次循环里，plb都是同一类故障
            x1_temp = branch_num
            x2_temp = branch_state
            y_temp = plb
            x1 = np.vstack([x1, x1_temp]) # 垂直拼接，x1_train的尺寸为(样本数,时滞,模拟量个数)
            x2 = np.vstack([x2, x2_temp]) # 垂直拼接，x2_train的尺寸为(样本数,时滞,状态量个数)
            y = np.vstack([y, y_temp]) # 垂直拼接，y_train的尺寸为(样本数,故障类型个数)
        
        return x1, x2, y
    
    def __getitem__(self,index):
        return self.x_num[index], self.x_state[index], self.y[index]
    
    def __len__(self):
        return len(self.y)
    
    ## 数据形式转化 (batch,time_lag,varsdim)
    def __add_window__(self, time_series, time_lag, dtype):
        """
        time_series: 输入二维数组(N,D)，N为样本维度，D为参数维度
        time_lag: 代表时间窗，用到多少个历史数据
        dtype: 指定加窗后的数组类型

        series_window: 二维数组转化为三维 (sample,timelag,variable)
        """
        time_series = np.array(time_series, dtype=dtype)
        total_time = time_series.shape[0]
        vars_num = time_series.shape[1]
        series_window = np.zeros([total_time - time_lag + 1, time_lag, vars_num], dtype=dtype)

        for i in range(series_window.shape[0]):
            series_window[i, :, :] = time_series[i:i + time_lag, :]

        return series_window
    
    ## 独热编码
    def __to_categorical__(self, y, num_classes=None):
        """
        y: 独热编码对象 需要数据为int类型
        num_classes: 独热编码到的维度
        """
        y = np.array(y, dtype=np.int32)
        input_shape = y.shape
        if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
            input_shape = tuple(input_shape[:-1])
        y = y.ravel()
        if not num_classes:
            num_classes = np.max(y) + 1
        n = y.shape[0]
        categorical = np.zeros((n, num_classes), dtype=np.float32)
        categorical[np.arange(n), y] = 1
        output_shape = input_shape + (num_classes,)
        categorical = np.reshape(categorical, output_shape)
        return categorical

    ## 划分数据集
    def __split_data__(self, data1, data2, label, per):
        """
        data_1: 三维模拟量数据
        data_2: 三维状态量数据
        label: 样本标签
        per: 随机划分的训练集占百分比

        x1_train: 三维，训练数据的模拟量部分
        x2_train: 三维，训练数据的状态量部分
        y_train: 二维，训练数据的标签
        x1_valid: 三维，验证数据的模拟量部分
        x2_valid: 三维，验证数据的状态量部分
        y_valid: 二维，验证数据的标签
        """
        # 划分规则为随机划分60作为训练，40作为验证
        np.random.seed(0)
        num_samples = len(label)
        rank = np.arange(num_samples)
        np.random.shuffle(rank)
        rank_train = rank[:int(per * num_samples)]
        rank_valid = rank[int(per * num_samples):]

        # 正式划分数据
        x1_train = data1[rank_train].copy()
        x2_train = data2[rank_train].copy()
        y_train = label[rank_train].copy()
        x1_valid = data1[rank_valid].copy()
        x2_valid = data2[rank_valid].copy()
        y_valid = label[rank_valid].copy()

        return x1_train, x2_train, y_train, x1_valid, x2_valid, y_valid
    
    ## 数据标准化 对训练和验证数据
    def __data_normalization__(self, data_train, data_valid):
        """
        data_train: 训练集，三维，模拟量数据
        data_valid: 验证集，三维，模拟量数据

        data_mean: 模拟量数据的均值
        data_std: 模拟量数据的标准差
        x_train_norm: 标准化后的训练数据，模拟量
        x_valid_norm: 标准化后的验证数据，模拟量
        """
        x_train_reshape = data_train.reshape([-1, data_train.shape[-1]])
        data_train_unique = np.unique(x_train_reshape, axis=0)
        data_mean = np.mean(data_train_unique, 0)
        data_std = np.std(data_train_unique, 0, ddof=1)
        data_std[np.where(abs(data_std) <= 1e-10)] = 1

        x_train_norm = ((x_train_reshape - data_mean) / data_std).reshape(data_train.shape)
        x_valid_reshape = data_valid.reshape([-1, data_valid.shape[-1]])
        x_valid_norm = ((x_valid_reshape - data_mean) / data_std).reshape(data_valid.shape)

        return data_mean, data_std, x_train_norm, x_valid_norm
    

    ## 待修改！！！
    def create_batch(self, batchsz): # 创建batch，即创建batchsz个任务，每个任务包括支持集和查询集，类别标签是随机选择的，图片是随机选择的，图片文件名存储在self.support_x_batch和self.query_x_batch中
        """
        create batch for meta-learning.
        ×episode× here means batch, and it means how many sets we want to retain.
        :param episodes: batch size
        :return:
        """
        self.support_x_batch = []  # support set batch # 用于存储batchsz个任务的支持集
        self.query_x_batch = []  # query set batch # 用于存储batchsz个任务的查询集
        for b in range(batchsz):  # for each batch
            # 1.select n_way classes randomly
            selected_cls = np.random.choice(self.cls_num, self.n_way, False)  # no duplicate # 对于每一个任务随机选择n_way个类别，False表示不可重复
            np.random.shuffle(selected_cls) # 打乱类别的顺序
            support_x = [] # 用于存储当前任务的支持集
            query_x = [] # 用于存储当前任务的查询集
            for cls in selected_cls: # 遍历每个选择出的类别
                # 2. select k_shot + k_query for each class
                selected_imgs_idx = np.random.choice(len(self.data[cls]), self.k_shot + self.k_query, False) # 对于每个任务的每个类别随机选择k_shot + k_query个图片，False表示不可重复
                np.random.shuffle(selected_imgs_idx) # 打乱图片的顺序
                indexDtrain = np.array(selected_imgs_idx[:self.k_shot])  # idx for Dtrain # 前面k_shot个图片的索引作为支持集的索引
                indexDtest = np.array(selected_imgs_idx[self.k_shot:])  # idx for Dtest # 从k_shot开始，后面k_query个图片的索引作为查询集的索引
                support_x.append(
                    np.array(self.data[cls])[indexDtrain].tolist())  # get all images filename for current Dtrain # 从data的当前类别cls中取出支持集的图片的文件名
                query_x.append(np.array(self.data[cls])[indexDtest].tolist()) # 从data的当前类别cls中取出查询集的图片的文件名，这个是最内层的列表，即每个类别对应一个列表，列表中存储该类别的所有图片的路径

            # shuffle the correponding relation between support set and query set
            random.shuffle(support_x) # 打乱支持集的顺序，这个是中间层的列表，即每个任务对应一个列表，列表中存储该任务的所有类别的支持集的图片的文件名
            random.shuffle(query_x) # 打乱查询集的顺序，这个是中间层的列表，即每个任务对应一个列表，列表中存储该任务的所有类别的查询集的图片的文件名

            self.support_x_batch.append(support_x)  # append set to current sets # 将当前任务的支持集添加到support_x_batch中，这个是最外层的列表，即存储batchsz个任务的支持集
            self.query_x_batch.append(query_x)  # append sets to current sets # 将当前任务的查询集添加到query_x_batch中，这个是最外层的列表，即存储batchsz个任务的查询集



    ## 数据标准化 仅对训练数据
    def __full_data_normalization__(self, data_train):
        """
        data_train: 训练集，三维，模拟量数据
        data_valid: 验证集，三维，模拟量数据

        data_mean: 模拟量数据的均值
        data_std: 模拟量数据的标准差
        x_train_norm: 标准化后的训练数据，模拟量
        x_valid_norm: 标准化后的验证数据，模拟量
        """
        x_train_reshape = data_train.reshape([-1, data_train.shape[-1]])
        data_train_unique = np.unique(x_train_reshape, axis=0)
        data_mean = np.mean(data_train_unique, 0)
        data_std = np.std(data_train_unique, 0, ddof=1)
        data_std[np.where(abs(data_std) <= 1e-10)] = 1

        x_train_norm = ((x_train_reshape - data_mean) / data_std).reshape(data_train.shape)

        return data_mean, data_std, x_train_norm
####################################################################################################################





####################################################################################################################
def main():
    root = os.path.join(os.getcwd(), 'xwdata')
    seq_len = 10
    dl = DataLoader(xwdataset(root = root, mode = 'train', twindow = seq_len, batchsz = 5, n_way = 5, k_shot = 1, k_query = 15, startidx=0), batch_size=5, shuffle=True)
    for index, (x_num, x_state, y) in enumerate(dl):
        print('index:', index)
        print('x_num shape:', x_num.shape)
        print('x_state shape :', x_state.shape)
        print('y:', y.shape)
        break
        

if __name__ == '__main__':
    main()