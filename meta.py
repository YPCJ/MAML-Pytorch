import  torch
from    torch import nn
from    torch import optim
from    torch.nn import functional as F
from    torch.utils.data import TensorDataset, DataLoader
from    torch import optim
import  numpy as np

from    learner import Learner
from    copy import deepcopy



class Meta(nn.Module):
    """
    Meta Learner
    """
    def __init__(self, args, config):
        """

        :param args:
        """
        super(Meta, self).__init__() # 调用父类的构造函数

        self.update_lr = args.update_lr
        self.meta_lr = args.meta_lr
        self.n_way = args.n_way
        self.k_spt = args.k_spt
        self.k_qry = args.k_qry
        self.task_num = args.task_num
        self.update_step = args.update_step
        self.update_step_test = args.update_step_test


        self.net = Learner(config, args.imgc, args.imgsz) # 实例化Learner类，传入网络结构、输入通道数、输入图像大小
        self.meta_optim = optim.Adam(self.net.parameters(), lr=self.meta_lr) # 定义优化器，使用Adam优化器，学习率为meta_lr




    def clip_grad_by_norm_(self, grad, max_norm): # 定义梯度裁剪函数,这里的梯度裁剪是在原地进行的,grad是梯度，max_norm是最大范数
        """
        in-place gradient clipping.
        :param grad: list of gradients
        :param max_norm: maximum norm allowable
        :return:
        """

        total_norm = 0
        counter = 0
        for g in grad: # 遍历梯度
            param_norm = g.data.norm(2) # 计算梯度的范数,2表示二范数
            total_norm += param_norm.item() ** 2 # 累加梯度的范数的平方
            counter += 1
        total_norm = total_norm ** (1. / 2) # 计算梯度的范数,1./2表示开方

        clip_coef = max_norm / (total_norm + 1e-6) # 计算裁剪系数,1e-6是为了防止除0
        if clip_coef < 1: # 如果裁剪系数小于1,则进行裁剪
            for g in grad:
                g.data.mul_(clip_coef) # 将梯度乘以裁剪系数

        return total_norm/counter # 返回梯度的范数的平均值


    def forward(self, x_spt, y_spt, x_qry, y_qry): # 前向传播,这里的x_spt是支持集的输入，y_spt是支持集的标签，x_qry是查询集的输入，y_qry是查询集的标签
        """

        :param x_spt:   [b, setsz, c_, h, w] # b是任务数 setsz是支持集的样本数
        :param y_spt:   [b, setsz]
        :param x_qry:   [b, querysz, c_, h, w] # b是任务数 querysz是查询集的样本数
        :param y_qry:   [b, querysz] 
        :return:
        """
        task_num, setsz, c_, h, w = x_spt.size() # 获取支持集的样本数，输入通道数，输入图像的高和宽
        querysz = x_qry.size(1) # 获取查询集的样本数

        losses_q = [0 for _ in range(self.update_step + 1)]  # losses_q[i] is the loss on step i # 用于存储每一步的损失,这里的损失是在任务内计算的
        corrects = [0 for _ in range(self.update_step + 1)]  # 用于存储每一步的准确率,这里的准确率是在任务内计算的


        for i in range(task_num): # 遍历每个任务

            # 1. run the i-th task and compute loss for k=0 # 运行第i个任务并计算k=0时的损失
            logits = self.net(x_spt[i], vars=None, bn_training=True) # 计算支持集的输出,这里的vars=None表示使用初始的参数，bn_training=True表示要训练bn层
            loss = F.cross_entropy(logits, y_spt[i]) # 计算损失
            grad = torch.autograd.grad(loss, self.net.parameters()) # 计算梯度
            fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, self.net.parameters()))) 
            # 上面这步计算快速权重,这里的zip(grad, self.net.parameters())表示将梯度和参数打包成元组，然后使用map函数对每个元组进行操作，
            # 这里的操作是p[1] - self.update_lr * p[0]，p[1]表示参数，p[0]表示梯度，self.update_lr表示学习率，最后使用list函数将map函数的结果转换为列表
            # 即通过一步梯度下降更新参数得到快速权重，这个快速权重将在后面的任务内更新中使用

            # this is the loss and accuracy before first update # 这是第一次更新之前的损失和准确率
            with torch.no_grad(): # 这里的with torch.no_grad()表示不进行梯度计算，因为这里的损失和准确率是在任务内计算的，不需要计算梯度
                # [setsz, nway]
                logits_q = self.net(x_qry[i], self.net.parameters(), bn_training=True) # 计算查询集的输出，这里的self.net.parameters()表示使用初始的参数，bn_training=True表示要训练bn层
                loss_q = F.cross_entropy(logits_q, y_qry[i]) # 计算损失,这里的损失是在任务内计算的,使用cross_entropy函数计算交叉熵损失
                losses_q[0] += loss_q # 累加损失

                pred_q = F.softmax(logits_q, dim=1).argmax(dim=1) # 计算预测值，dim=1表示按行计算，argmax(dim=1)表示取每行的最大值的索引，logits_q是查询集的输出，使用softmax函数计算概率，然后取最大概率的索引
                correct = torch.eq(pred_q, y_qry[i]).sum().item() # 计算准确率，torch.eq(pred_q, y_qry[i])表示比较预测值和标签，相等的返回True，不相等的返回False，然后使用sum函数计算True的个数，最后使用item函数将结果转换为标量
                corrects[0] = corrects[0] + correct # 累加准确率，corrects[0]表示第一次更新之前的准确率

            # this is the loss and accuracy after the first update # 这是第一次更新之后的损失和准确率
            with torch.no_grad():
                # [setsz, nway]
                logits_q = self.net(x_qry[i], fast_weights, bn_training=True) # 计算查询集的输出，这里的fast_weights表示使用快速权重，bn_training=True表示要训练bn层
                loss_q = F.cross_entropy(logits_q, y_qry[i]) # 计算损失,这里的损失是在任务内计算的,使用cross_entropy函数计算交叉熵损失
                losses_q[1] += loss_q # 累加损失
                # [setsz]
                pred_q = F.softmax(logits_q, dim=1).argmax(dim=1) # 计算预测值，dim=1表示按行计算，argmax(dim=1)表示取每行的最大值的索引，logits_q是查询集的输出，使用softmax函数计算概率，然后取最大概率的索引
                correct = torch.eq(pred_q, y_qry[i]).sum().item() # 计算准确率，torch.eq(pred_q, y_qry[i])表示比较预测值和标签，相等的返回True，不相等的返回False，然后使用sum函数计算True的个数，最后使用item函数将结果转换为标量
                corrects[1] = corrects[1] + correct # 累加准确率，corrects[1]表示第一次更新之后的准确率

            for k in range(1, self.update_step): # 遍历每一步更新
                
                # 1. run the i-th task and compute loss for k=1~K-1 # 运行第i个任务并计算k=1~K-1时的损失
                logits = self.net(x_spt[i], fast_weights, bn_training=True) # 计算支持集的输出,这里的fast_weights表示使用快速权重，bn_training=True表示要训练bn层
                loss = F.cross_entropy(logits, y_spt[i]) # 计算损失,这里的损失是在任务内计算的,使用cross_entropy函数计算交叉熵损失
                
                # 2. compute grad on theta_pi # 计算theta_pi上的梯度,这里的theta_pi表示快速权重
                grad = torch.autograd.grad(loss, fast_weights) # 计算梯度,使用autograd.grad函数计算梯度
                
                # 3. theta_pi = theta_pi - train_lr * grad
                fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights))) 
                # 更新快速权重,这里的zip(grad, fast_weights)表示将梯度和快速权重打包成元组，然后使用map函数对每个元组进行操作
                # 这里的操作是p[1] - self.update_lr * p[0]，p[1]表示快速权重，p[0]表示梯度，self.update_lr表示学习率，最后使用list函数将map函数的结果转换为列表
                # 即通过一步梯度下降更新快速权重,一共要更新self.update_step步，注意这里在更新时使用的是支持集的数据，而不是查询集的数据，记录了self.update_step步的梯度

                # 计算第k+1步的损失和准确率，loss要记录grad，准确率不需要记录grad
                logits_q = self.net(x_qry[i], fast_weights, bn_training=True)
                # loss_q will be overwritten and just keep the loss_q on last update step. # 这里的loss_q将被覆盖，只保留最后一步的loss_q
                loss_q = F.cross_entropy(logits_q, y_qry[i]) # 计算损失,这里的损失是在任务内计算的,使用cross_entropy函数计算交叉熵损失
                losses_q[k + 1] += loss_q # 累加损失，losses_q[k + 1]表示第k+1步的损失，k从1开始，所以累加的损失是从第二步开始的

                with torch.no_grad():
                    pred_q = F.softmax(logits_q, dim=1).argmax(dim=1) # 计算预测值，dim=1表示按行计算，argmax(dim=1)表示取每行的最大值的索引，logits_q是查询集的输出，使用softmax函数计算概率，然后取最大概率的索引
                    correct = torch.eq(pred_q, y_qry[i]).sum().item()  # convert to numpy # 计算准确率，torch.eq(pred_q, y_qry[i])表示比较预测值和标签，相等的返回True，不相等的返回False，然后使用sum函数计算True的个数，最后使用item函数将结果转换为标量
                    corrects[k + 1] = corrects[k + 1] + correct # 累加准确率，corrects[k + 1]表示第k+1步的准确率，k从1开始，所以累加的准确率是从第二步开始的



        # end of all tasks # 所有任务结束
        # sum over all losses on query set across all tasks # 对所有任务的查询集上的损失求和
        loss_q = losses_q[-1] / task_num 
        # 计算平均损失，losses_q[-1]表示最后一步的损失


        # optimize theta parameters
        self.meta_optim.zero_grad() # 梯度清零
        loss_q.backward() # 反向传播, 计算梯度, 用最后一步的损失计算梯度可以减少计算量
        # print('meta update')
        # for p in self.net.parameters()[:5]:
        # 	print(torch.norm(p).item())
        self.meta_optim.step() # 更新参数,这里的参数是theta,即网络的参数,这里的更新是在所有任务上计算的梯度的基础上进行的,是MAML中的先验知识，即所定义网络的起始权重


        accs = np.array(corrects) / (querysz * task_num) # 计算准确率，用累加的正确个数corrects除以查询集的样本数querysz乘以任务数task_num，得到平均准确率

        return accs # 返回准确率


    def finetunning(self, x_spt, y_spt, x_qry, y_qry):
        """

        :param x_spt:   [setsz, c_, h, w]
        :param y_spt:   [setsz]
        :param x_qry:   [querysz, c_, h, w]
        :param y_qry:   [querysz]
        :return:
        """
        assert len(x_spt.shape) == 4 # 判断x_spt的维度是否为4，因为只取一个任务，所以x_spt的维度为4，如果是5，即sample出了多个任务，则会报错

        querysz = x_qry.size(0) # 获取查询集的样本数

        corrects = [0 for _ in range(self.update_step_test + 1)] # 用于存储每一步的准确率,这里的准确率是在任务内计算的

        # in order to not ruin the state of running_mean/variance and bn_weight/bias # 为了不破坏running_mean/variance和bn_weight/bias的状态
        # we finetunning on the copied model instead of self.net # 我们在复制的模型上进行微调，而不是在self.net上进行微调
        net = deepcopy(self.net) # 复制网络，这里的复制是深拷贝，即复制的是值，而不是引用，改变复制的网络不会改变原网络

        # 1. run the i-th task and compute loss for k=0
        logits = net(x_spt) # 计算支持集的输出
        loss = F.cross_entropy(logits, y_spt) # 计算损失
        grad = torch.autograd.grad(loss, net.parameters()) # 计算梯度
        fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, net.parameters()))) 
        # 计算快速权重，这里的zip(grad, net.parameters())表示将梯度和参数打包成元组，然后使用map函数对每个元组进行操作
        # 这里的操作是p[1] - self.update_lr * p[0]，p[1]表示参数，p[0]表示梯度，self.update_lr表示学习率，最后使用list函数将map函数的结果转换为列表

        # this is the loss and accuracy before first update
        with torch.no_grad():
            # [setsz, nway]
            logits_q = net(x_qry, net.parameters(), bn_training=True)
            # [setsz]
            pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
            # scalar
            correct = torch.eq(pred_q, y_qry).sum().item()
            corrects[0] = corrects[0] + correct

        # this is the loss and accuracy after the first update
        with torch.no_grad():
            # [setsz, nway]
            logits_q = net(x_qry, fast_weights, bn_training=True)
            # [setsz]
            pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
            # scalar
            correct = torch.eq(pred_q, y_qry).sum().item()
            corrects[1] = corrects[1] + correct

        for k in range(1, self.update_step_test):
            # 1. run the i-th task and compute loss for k=1~K-1
            logits = net(x_spt, fast_weights, bn_training=True)
            loss = F.cross_entropy(logits, y_spt)
            # 2. compute grad on theta_pi
            grad = torch.autograd.grad(loss, fast_weights)
            # 3. theta_pi = theta_pi - train_lr * grad
            fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))

            logits_q = net(x_qry, fast_weights, bn_training=True)
            # loss_q will be overwritten and just keep the loss_q on last update step.
            loss_q = F.cross_entropy(logits_q, y_qry)

            with torch.no_grad():
                pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                correct = torch.eq(pred_q, y_qry).sum().item()  # convert to numpy
                corrects[k + 1] = corrects[k + 1] + correct


        del net # 删除复制的网络，不更新起始权重，因为这个函数只是用于测试，不会更新self.net的参数

        accs = np.array(corrects) / querysz # 计算准确率，用累加的正确个数corrects除以查询集的样本数querysz，得到平均准确率

        return accs




def main():
    pass 


if __name__ == '__main__':
    main()
