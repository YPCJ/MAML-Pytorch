import os
import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms
import numpy as np
import collections
from PIL import Image
import csv
import random
import pandas as pd

    
class xwdataset(Dateset):
    def __init__(self, root, mode, twindow, batchsz, n_way, k_shot, k_query, startidx=0):
        ####################################################################################################################
        # 设定整个数据集的路径
        # 获取当前运行文件路径
        print('root:', root)
        self.path = os.path.join(root, 'data')
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
        float_vars = pd.Series(num_vars + [np.nan] * (idx - len(num_vars))) # 用nan补齐
        int_vars = pd.Series(state_vars + [np.nan] * (idx - len(state_vars)))
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
            self.fault_dic[all_fault[i]] = i
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
        ## 读取数据集
        self.data, self.label = self.load_data(os.path.join(self.path)) # 读取csv文件,不是单独一个csv，因为里面还包含参数名文件来指示读取哪些连续量和状态量
        # 类别数是self.label中不同元素的个数
        self.num_classes = len(set(self.label)) # set()函数创建一个无序不重复元素集
        
        self.create_batch(self.batchsz)
        
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
        
        x1 = np.empty((0, time_lag, data_num.shape[-1]), dtype=np.float32)
        x2 = np.empty((0, time_lag, data_state.shape[-1]), dtype=np.int32)
        y = np.empty((0, len(all_fault)), dtype=np.int32)
        
        for i in range(np.max(file_num)):
            position = (file_num == i) # 选出第i个文件的数据，每个文件对应了一个故障类型，即一个标签，所以每个文件的数据都是同一类故障
            branch_num = self.add_window_(data_num[position], time_lag, dtype=np.float32)  # 加时滞窗，升维度
            branch_state = self.add_window_(data_state[position], time_lag, dtype=np.int32)  # 加时滞窗，升维度
            plb = np.array(data_label[position], dtype=np.int32)[time_lag - 1:] # 标签
            plb = self.to_categorical(plb, len(all_fault)) # 独热编码，这里其实是多分类问题，所以标签是独热编码，但其实在每次循环里，plb都是同一类故障
            x1_temp = branch_num
            x2_temp = branch_state
            y_temp = plb
            x1_train = np.vstack([x1_train, x1_train_temp]) # 垂直拼接，x1_train的尺寸为(样本数,时滞,模拟量个数)
            x2_train = np.vstack([x2_train, x2_train_temp]) # 垂直拼接，x2_train的尺寸为(样本数,时滞,状态量个数)
            y_train = np.vstack([y_train, y_train_temp]) # 垂直拼接，y_train的尺寸为(样本数,故障类型个数)
            x1_valid = np.vstack([x1_valid, x1_valid_temp]) # 垂直拼接，x1_valid的尺寸为(样本数,时滞,模拟量个数)
            x2_valid = np.vstack([x2_valid, x2_valid_temp]) # 垂直拼接，x2_valid的尺寸为(样本数,时滞,状态量个数)
            y_valid = np.vstack([y_valid, y_valid_temp]) # 垂直拼接，y_valid的尺寸为(样本数,故障类型个数)
        return 