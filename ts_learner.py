import  torch
from    torch import nn
from    torch.nn import functional as F
import  numpy as np



class ts_Learner(nn.Module):
    """

    """

    def __init__(self, config, time_windows, var_num):
        """

        :param config: network config file, type:list of (string, list)
        :param imgc: 1 or 3
        :param imgsz:  28 or 84
        """
        super(Learner, self).__init__() # 调用父类的构造函数


        self.config = config # 网络结构

        # this dict contains all tensors needed to be optimized
        self.vars = nn.ParameterList() # 用于存储需要优化的参数
        # running_mean and running_var
        self.vars_bn = nn.ParameterList() # 用于存储批标准化层的参数

        for i, (name, param) in enumerate(self.config): # 遍历网络结构
            if name == 'conv2d':
                # [ch_out, ch_in, kernelsz, kernelsz]
                w = nn.Parameter(torch.ones(*param[:4])) # 定义卷积核参数，将param[:4]传入，*param[:4]表示将param[:4]解包
                # gain=1 according to cbfin's implementation
                torch.nn.init.kaiming_normal_(w) # 使用kaiming_normal_初始化卷积核参数
                self.vars.append(w) # 将卷积核参数添加到vars中
                # [ch_out] # 定义偏置参数
                self.vars.append(nn.Parameter(torch.zeros(param[0]))) # 将偏置参数添加到vars中

            elif name == 'convt2d': # 转置卷积层
                # [ch_in, ch_out, kernelsz, kernelsz, stride, padding]
                w = nn.Parameter(torch.ones(*param[:4]))
                # gain=1 according to cbfin's implementation
                torch.nn.init.kaiming_normal_(w)
                self.vars.append(w)
                # [ch_in, ch_out]
                self.vars.append(nn.Parameter(torch.zeros(param[1])))

            elif name == 'lstm': # LSTM层
                # [input_size, hidden_size, num_layers]
                h0 = nn.Parameter(torch.zeros(param[2],1,param[1])) # 定义h0参数
                c0 = nn.Parameter(torch.zeros(param[2],1,param[1])) # 定义c0参数
                w = nn.Parameter(torch.ones(*param[:4]))
                # gain=1 according to cbfin's implementation
                torch.nn.init.kaiming_normal_(h0)
                torch.nn.init.kaiming_normal_(c0)
                torch.nn.init.kaiming_normal_(w) # 使用kaiming_normal_初始化卷积核参数
                self.vars.append(w)
                # [ch_in, ch_out]
                self.vars.append(nn.Parameter(torch.zeros(param[1])))

            elif name == 'linear':
                # [ch_out, ch_in]
                w = nn.Parameter(torch.ones(*param))
                # gain=1 according to cbfinn's implementation
                torch.nn.init.kaiming_normal_(w)
                self.vars.append(w)
                # [ch_out]
                self.vars.append(nn.Parameter(torch.zeros(param[0])))

            elif name == 'bn': # 批标准化层
                # [ch_out]
                w = nn.Parameter(torch.ones(param[0])) # 定义gamma参数
                self.vars.append(w) # 将gamma参数添加到vars中
                # [ch_out]
                self.vars.append(nn.Parameter(torch.zeros(param[0]))) # 定义beta参数，将beta参数添加到vars中

                # must set requires_grad=False
                running_mean = nn.Parameter(torch.zeros(param[0]), requires_grad=False) # 定义running_mean参数,requires_grad=False表示不需要计算梯度,因为这个参数是在训练过程中不断更新的
                running_var = nn.Parameter(torch.ones(param[0]), requires_grad=False) # 定义running_var参数,requires_grad=False表示不需要计算梯度,因为这个参数是在训练过程中不断更新的
                self.vars_bn.extend([running_mean, running_var]) # 将running_mean和running_var添加到vars_bn中


            elif name in ['tanh', 'relu', 'upsample', 'avg_pool2d', 'max_pool2d',
                          'flatten', 'reshape', 'leakyrelu', 'sigmoid']:
                continue
            else:
                raise NotImplementedError # 未实现错误






    def extra_repr(self):
        info = ''

        for name, param in self.config:
            if name == 'conv2d':
                tmp = 'conv2d:(ch_in:%d, ch_out:%d, k:%dx%d, stride:%d, padding:%d)'\
                      %(param[1], param[0], param[2], param[3], param[4], param[5],)
                info += tmp + '\n'

            elif name == 'convt2d': # 转置卷积层
                tmp = 'convTranspose2d:(ch_in:%d, ch_out:%d, k:%dx%d, stride:%d, padding:%d)'\
                      %(param[0], param[1], param[2], param[3], param[4], param[5],)
                info += tmp + '\n'

            elif name == 'lstm': # LSTM层
                tmp = 'lstm:(ch_in:%d, ch_out:%d)'%(param[0], param[1])
                info += tmp + '\n'

            elif name == 'linear': # 全连接层
                tmp = 'linear:(in:%d, out:%d)'%(param[1], param[0])
                info += tmp + '\n'

            elif name == 'leakyrelu': # leakyrelu激活函数
                tmp = 'leakyrelu:(slope:%f)'%(param[0])
                info += tmp + '\n'


            elif name == 'avg_pool2d':
                tmp = 'avg_pool2d:(k:%d, stride:%d, padding:%d)'%(param[0], param[1], param[2])
                info += tmp + '\n'
            elif name == 'max_pool2d':
                tmp = 'max_pool2d:(k:%d, stride:%d, padding:%d)'%(param[0], param[1], param[2])
                info += tmp + '\n'
            elif name in ['flatten', 'tanh', 'relu', 'upsample', 'reshape', 'sigmoid', 'use_logits', 'bn']:
                tmp = name + ':' + str(tuple(param))
                info += tmp + '\n'
            else:
                raise NotImplementedError

        return info



    def forward(self, x, vars=None, bn_training=True):
        """
        This function can be called by finetunning, however, in finetunning, we dont wish to update
        running_mean/running_var. Thought weights/bias of bn is updated, it has been separated by fast_weights.
        Indeed, to not update running_mean/running_var, we need set update_bn_statistics=False
        but weight/bias will be updated and not dirty initial theta parameters via fast_weiths.
        :param x: [b, 1, 28, 28]
        :param vars:
        :param bn_training: set False to not update
        :return: x, loss, likelihood, kld
        """

        if vars is None:
            vars = self.vars

        idx = 0
        bn_idx = 0

        for name, param in self.config:
            if name == 'conv2d':
                w, b = vars[idx], vars[idx + 1]
                # remember to keep synchrozied of forward_encoder and forward_decoder!
                x = F.conv2d(x, w, b, stride=param[4], padding=param[5])
                idx += 2
                # print(name, param, '\tout:', x.shape)
            elif name == 'convt2d':
                w, b = vars[idx], vars[idx + 1]
                # remember to keep synchrozied of forward_encoder and forward_decoder!
                x = F.conv_transpose2d(x, w, b, stride=param[4], padding=param[5])
                idx += 2
                # print(name, param, '\tout:', x.shape)
            elif name == 'lstm':
                w, b = vars[idx], vars[idx + 1]
                x = F.lstm(x, w, b, stride=param[4], padding=param[5])
                idx += 2
                # print(name, param, '\tout:', x.shape)
            elif name == 'linear':
                w, b = vars[idx], vars[idx + 1]
                x = F.linear(x, w, b)
                idx += 2
                # print('forward:', idx, x.norm().item())
            elif name == 'bn':
                w, b = vars[idx], vars[idx + 1]
                running_mean, running_var = self.vars_bn[bn_idx], self.vars_bn[bn_idx+1]
                x = F.batch_norm(x, running_mean, running_var, weight=w, bias=b, training=bn_training)
                idx += 2
                bn_idx += 2

            elif name == 'flatten':
                # print(x.shape)
                x = x.view(x.size(0), -1)
            elif name == 'reshape':
                # [b, 8] => [b, 2, 2, 2]
                x = x.view(x.size(0), *param)
            elif name == 'relu':
                x = F.relu(x, inplace=param[0])
            elif name == 'leakyrelu':
                x = F.leaky_relu(x, negative_slope=param[0], inplace=param[1])
            elif name == 'tanh':
                x = F.tanh(x)
            elif name == 'sigmoid':
                x = torch.sigmoid(x)
            elif name == 'upsample':
                x = F.upsample_nearest(x, scale_factor=param[0])
            elif name == 'max_pool2d':
                x = F.max_pool2d(x, param[0], param[1], param[2])
            elif name == 'avg_pool2d':
                x = F.avg_pool2d(x, param[0], param[1], param[2])

            else:
                raise NotImplementedError

        # make sure variable is used properly
        assert idx == len(vars) # 判断是否遍历完vars
        assert bn_idx == len(self.vars_bn) # 判断是否遍历完vars_bn


        return x


    def zero_grad(self, vars=None): # 将梯度置零
        """

        :param vars:
        :return:
        """
        with torch.no_grad():
            if vars == None:
                for p in self.vars:
                    if p.grad is not None:
                        p.grad.zero_()
            else:
                for p in vars:
                    if p.grad is not None:
                        p.grad.zero_()

    def parameters(self):
        """
        override this function since initial parameters will return with a generator.
        :return:
        """
        return self.vars