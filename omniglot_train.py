import  torch, os
import  numpy as np
from    omniglotNShot import OmniglotNShot
import  argparse

from    meta import Meta
import time

def main(args):

    torch.manual_seed(222)
    torch.cuda.manual_seed_all(222)
    np.random.seed(222)

    print(args)

    config = [
        ('conv2d', [64, 1, 3, 3, 2, 0]),
        ('relu', [True]),
        ('bn', [64]),
        ('conv2d', [64, 64, 3, 3, 2, 0]),
        ('relu', [True]),
        ('bn', [64]),
        ('conv2d', [64, 64, 3, 3, 2, 0]),
        ('relu', [True]),
        ('bn', [64]),
        ('conv2d', [64, 64, 2, 2, 1, 0]),
        ('relu', [True]),
        ('bn', [64]),
        ('flatten', []),
        ('linear', [args.n_way, 64])
    ]

    ## 设定设备
    if torch.cuda.is_available(): # 判断是否有GPU
        device = torch.device('cuda:0') # 指定设备 cuda
    elif torch.backends.mps.is_available(): # 判断是否有MPS
        device = torch.device('mps') # 指定设备 mps
    else:
        device = torch.device('cpu') # 指定设备 cpu
    
    maml = Meta(args, config).to(device)

    tmp = filter(lambda x: x.requires_grad, maml.parameters())
    num = sum(map(lambda x: np.prod(x.shape), tmp))
    print(maml)
    print('Total trainable tensors:', num)

    db_train = OmniglotNShot('omniglot',
                       batchsz=args.task_num,
                       n_way=args.n_way,
                       k_shot=args.k_spt,
                       k_query=args.k_qry,
                       imgsz=args.imgsz)
    
    print('Start training...')
    ori_time = time.time() # 记录开始时间
    for step in range(args.epoch):
        if step == 0:
            start_time = time.time() # 记录开始100轮的时间
        x_spt, y_spt, x_qry, y_qry = db_train.next()
        x_spt, y_spt, x_qry, y_qry = torch.from_numpy(x_spt).to(device), torch.from_numpy(y_spt).to(device), \
                                     torch.from_numpy(x_qry).to(device), torch.from_numpy(y_qry).to(device)

        # set traning=True to update running_mean, running_variance, bn_weights, bn_bias
        accs = maml(x_spt, y_spt, x_qry, y_qry)

        if step % 100 == 99:
            end_time = time.time() # 记录结束100轮的时间
            print('step:', step+1, '\ttraining acc:', accs, '\ttime:{:.4f}s'.format(end_time - start_time))
            start_time = time.time() # 记录开始100轮的时间
        if step % 500 == 499:
            start_test_time = time.time() # 记录开始测试的时间
            accs = []
            for _ in range(1000//args.task_num):
                # test
                x_spt, y_spt, x_qry, y_qry = db_train.next('test')
                x_spt, y_spt, x_qry, y_qry = torch.from_numpy(x_spt).to(device), torch.from_numpy(y_spt).to(device), \
                                             torch.from_numpy(x_qry).to(device), torch.from_numpy(y_qry).to(device)

                # split to single task each time
                for x_spt_one, y_spt_one, x_qry_one, y_qry_one in zip(x_spt, y_spt, x_qry, y_qry):
                    test_acc = maml.finetunning(x_spt_one, y_spt_one, x_qry_one, y_qry_one)
                    accs.append( test_acc )

            # [b, update_step+1]
            accs = np.array(accs).mean(axis=0).astype(np.float16)
            end_test_time = time.time() # 记录结束测试的时间
            print('Test step:', (step+1)//500,'\tacc:', accs,'\ttest time:{:.4f}s'.format(end_time_test-start_time_test))
            start_time = time.time() # 记录开始100轮的时间
    
    ## 测试最终模型
    print('Testing...')
    for _ in range(1000//args.task_num):
        # test
        x_spt, y_spt, x_qry, y_qry = db_train.next('test')
        x_spt, y_spt, x_qry, y_qry = torch.from_numpy(x_spt).to(device), torch.from_numpy(y_spt).to(device), \
                                     torch.from_numpy(x_qry).to(device), torch.from_numpy(y_qry).to(device)

        # split to single task each time
        for x_spt_one, y_spt_one, x_qry_one, y_qry_one in zip(x_spt, y_spt, x_qry, y_qry):
            test_acc = maml.finetunning(x_spt_one, y_spt_one, x_qry_one, y_qry_one)
            accs.append( test_acc )
    last_time = time.time() # 记录开始时间
    print('Final test:', '\tacc:', test_acc, '\tTotal time:{:.4f}s'.format(last_time - ori_time)) # 打印总时间

if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--epoch', type=int, help='epoch number', default=40000)
    argparser.add_argument('--n_way', type=int, help='n way', default=5)
    argparser.add_argument('--k_spt', type=int, help='k shot for support set', default=1)
    argparser.add_argument('--k_qry', type=int, help='k shot for query set', default=15)
    argparser.add_argument('--imgsz', type=int, help='imgsz', default=28)
    argparser.add_argument('--imgc', type=int, help='imgc', default=1)
    argparser.add_argument('--task_num', type=int, help='meta batch size, namely task num', default=32)
    argparser.add_argument('--meta_lr', type=float, help='meta-level outer learning rate', default=1e-3)
    argparser.add_argument('--update_lr', type=float, help='task-level inner update learning rate', default=0.4)
    argparser.add_argument('--update_step', type=int, help='task-level inner update steps', default=5)
    argparser.add_argument('--update_step_test', type=int, help='update steps for finetunning', default=10)

    args = argparser.parse_args()

    main(args)
