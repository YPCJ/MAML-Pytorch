import torch, os
import numpy as np
from MiniImagenet import MiniImagenet
import scipy.stats
from torch.utils.data import DataLoader
from    torch.optim import lr_scheduler
import  random, sys, pickle
import  argparse
import time

from meta import Meta


def mean_confidence_interval(accs, confidence=0.95): # 定义置信区间
    n = accs.shape[0] # 一共有多少个样本
    m, se = np.mean(accs), scipy.stats.sem(accs) # 计算均值和标准误差，sem表示标准误差
    h = se * scipy.stats.t._ppf((1 + confidence) / 2, n - 1) # 计算置信区间，t.ppf表示累积分布函数的反函数，通过置信度和自由度计算置信区间
    return m, h # 返回均值和置信区间


def main():

    torch.manual_seed(222) # 为CPU设置种子用于生成随机数，以使得结果是确定的
    torch.cuda.manual_seed_all(222) # 为所有GPU设置种子，以使得结果是确定的
    np.random.seed(222) # 为numpy设置种子，以使得结果是确定的

    print(args) # 打印参数

    config = [                              # 定义网络结构，包括卷积层、激活函数、批标准化层、最大池化层、全连接层
        ('conv2d', [32, 3, 3, 3, 1, 0]),    # 卷积层1，输入通道数为3，输出通道数为32，卷积核大小为3*3，步长为1，padding为0
        ('relu', [True]),                   # 激活函数1，relu
        ('bn', [32]),                       # 批标准化层1，输入通道数为32
        ('max_pool2d', [2, 2, 0]),          # 最大池化层1，池化核大小为2*2，步长为2，padding为0
        ('conv2d', [32, 32, 3, 3, 1, 0]),   # 卷积层2，输入通道数为32，输出通道数为32，卷积核大小为3*3，步长为1，padding为0
        ('relu', [True]),                   # 激活函数2，relu
        ('bn', [32]),                       # 批标准化层2，输入通道数为32
        ('max_pool2d', [2, 2, 0]),          # 最大池化层2，池化核大小为2*2，步长为2，padding为0
        ('conv2d', [32, 32, 3, 3, 1, 0]),   # 卷积层3，输入通道数为32，输出通道数为32，卷积核大小为3*3，步长为1，padding为0
        ('relu', [True]),                   # 激活函数3，relu
        ('bn', [32]),                       # 批标准化层3，输入通道数为32
        ('max_pool2d', [2, 2, 0]),          # 最大池化层3，池化核大小为2*2，步长为2，padding为0
        ('conv2d', [32, 32, 3, 3, 1, 0]),   # 卷积层4，输入通道数为32，输出通道数为32，卷积核大小为3*3，步长为1，padding为0
        ('relu', [True]),                   # 激活函数4，relu
        ('bn', [32]),                       # 批标准化层4，输入通道数为32
        ('max_pool2d', [2, 1, 0]),          # 最大池化层4，池化核大小为2*2，步长为1，padding为0
        ('flatten', []),                    # 展平层
        ('linear', [args.n_way, 32 * 5 * 5])# 全连接层，输入通道数为32*5*5，这个5*5是经过四次池化后的结果，输出通道数为n_way,由args指定
    ]

    device = torch.device('mps') # 指定设备 MPS
    maml = Meta(args, config).to(device) # 实例化Meta类，传入参数和网络结构，将网络放到MPS上

    tmp = filter(lambda x: x.requires_grad, maml.parameters()) # 过滤出需要梯度的参数
    num = sum(map(lambda x: np.prod(x.shape), tmp)) # 计算需要梯度的参数的个数
    print(maml) # 打印网络结构
    print('Total trainable tensors:', num) # 打印需要梯度的参数的个数

    # batchsz here means total episode number
    mini = MiniImagenet('./miniimagenet/', mode='train', n_way=args.n_way, k_shot=args.k_spt, # 实例化MiniImagenet类，传入参数
                        k_query=args.k_qry,
                        batchsz=10000, resize=args.imgsz)
    mini_test = MiniImagenet('./miniimagenet/', mode='test', n_way=args.n_way, k_shot=args.k_spt, # 实例化MiniImagenet类，传入参数
                             k_query=args.k_qry,
                             batchsz=100, resize=args.imgsz)
    
    print('Start training...')
    ori_time = time.time() # 记录开始时间
    for epoch in range(args.epoch//10000): # 训练epoch次,//表示整除,因为每次训练10000个任务，所以训练epoch次相当于训练epoch*10000个任务
        # fetch meta_batchsz num of episode each time
        # db = DataLoader(mini, args.task_num, shuffle=True, num_workers=1, pin_memory=True)
        print('epoch:', epoch+1)
        db = DataLoader(mini, args.task_num, shuffle=True, num_workers=0, pin_memory=True) # 每次从数据集中取出task_num个任务，包括支持集和查询集
        for step, (x_spt, y_spt, x_qry, y_qry) in enumerate(db): # 遍历每个任务
            
            if step == 0: # 第一个任务
                start_time = time.time() # 记录开始时间
            
            x_spt, y_spt, x_qry, y_qry = x_spt.to(device), y_spt.to(device), x_qry.to(device), y_qry.to(device) # 将数据放到device上

            accs = maml(x_spt, y_spt, x_qry, y_qry) # 计算准确率
            
            if step % 100 == 99: # 每100个任务打印一次准确率和时间
                end_time = time.time() # 记录结束时间
                print('step:', step+1, '\ttraining acc:', accs,'\ttime:{:.4f}s'.format(end_time - start_time)) # 打印准确率和时间
                start_time = time.time() # 重置时间
            if step % 500 == 499:  # evaluation # 每500个任务进行一次测试，测试时在任务内更新[update_step_test]步
                # db_test = DataLoader(mini_test, 1, shuffle=True, num_workers=1, pin_memory=True)
                start_time_test = time.time() # 记录开始时间
                db_test = DataLoader(mini_test, 1, shuffle=True, num_workers=0, pin_memory=True) # 每次从数据集中取出1个任务，包括支持集和查询集
                accs_all_test = [] # 用于存储每个任务的准确率

                for x_spt, y_spt, x_qry, y_qry in db_test: # 遍历每个任务
                    x_spt, y_spt, x_qry, y_qry = x_spt.squeeze(0).to(device), y_spt.squeeze(0).to(device), \
                                                 x_qry.squeeze(0).to(device), y_qry.squeeze(0).to(device)

                    accs = maml.finetunning(x_spt, y_spt, x_qry, y_qry) # 根据任务内的支持集和查询集更新base模型，返回准确率
                    accs_all_test.append(accs) # 将准确率添加到列表中

                # [b, update_step+1]
                accs = np.array(accs_all_test).mean(axis=0).astype(np.float16) # 计算所有任务的准确率的均值
                end_time_test = time.time() # 记录结束时间
                print('Test step:', (step+1)//500, 'Test acc:\t', accs,'\ttest time:{:.4f}s'.format(end_time_test-start_time_test)) # 打印准确率和时间
                start_time = time.time() # 重置时间
                
    print('Final:')
    db_test = DataLoader(mini_test, 1, shuffle=True, num_workers=0, pin_memory=True) # 每次从数据集中取出1个任务，包括支持集和查询集
    accs_all_test = [] # 用于存储每个任务的准确率

    for x_spt, y_spt, x_qry, y_qry in db_test: # 遍历每个任务
        x_spt, y_spt, x_qry, y_qry = x_spt.squeeze(0).to(device), y_spt.squeeze(0).to(device), \
                                        x_qry.squeeze(0).to(device), y_qry.squeeze(0).to(device)

        accs = maml.finetunning(x_spt, y_spt, x_qry, y_qry) # 根据任务内的支持集和查询集更新base模型，返回准确率
        accs_all_test.append(accs) # 将准确率添加到列表中

    # [b, update_step+1]
    accs = np.array(accs_all_test).mean(axis=0).astype(np.float16) # 计算所有任务的准确率的均值
    final_time = time.time() # 记录最终结束时间
    print('Test acc:\t', accs,'\ttotal time:{:.4f}s'.format(final_time-ori_time)) # 打印准确率和时间
    print ('Done!')

if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--epoch', type=int, help='epoch number', default=60000) # 训练的轮数
    argparser.add_argument('--n_way', type=int, help='n way', default=5) # 每个任务的类别数
    argparser.add_argument('--k_spt', type=int, help='k shot for support set', default=1) # 每个类别支持集的样本数
    argparser.add_argument('--k_qry', type=int, help='k shot for query set', default=15) # 每个类别查询集的样本数
    argparser.add_argument('--imgsz', type=int, help='imgsz', default=84) # 图片大小
    argparser.add_argument('--imgc', type=int, help='imgc', default=3) # 图片通道数
    argparser.add_argument('--task_num', type=int, help='meta batch size, namely task num', default=4) # 每个batch的任务数
    argparser.add_argument('--meta_lr', type=float, help='meta-level outer learning rate', default=1e-3) # 元学习率，外圈的学习率
    argparser.add_argument('--update_lr', type=float, help='task-level inner update learning rate', default=0.01) # 任务学习率，内圈的学习率
    argparser.add_argument('--update_step', type=int, help='task-level inner update steps', default=5) # 训练时在任务内更新的步数
    argparser.add_argument('--update_step_test', type=int, help='update steps for finetunning', default=10) # 测试时在任务内更新的步数

    args = argparser.parse_args() # 解析参数

    main()
