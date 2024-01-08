import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from xwdataset import xwdataset
import os
from tqdm import tqdm
import time

class ts_Learner(nn.Module):
## 时序学习器，用于时序数据的分类模型，包含模拟量输入层，状态量嵌入层和输入层，LSTM层，输出层
    def __init__(self, input_size_num, input_size_state, feature_size, hidden_size, num_cls,embedding_word = 4, embedding_dim = 4):
        super(ts_Learner, self).__init__()
        self.input_size_num = input_size_num
        self.input_size_state = input_size_state
        self.feature_size = feature_size
        self.embedding_dim = embedding_dim
        self.embedding_word = embedding_word
        self.hidden_size = hidden_size
        self.num_cls = num_cls
        self.training = True

        self.vars = nn.ParameterList()
        
        # 模拟量输入层
        num_input_module_size = [input_size_num] + feature_size
        for i in range(1, len(num_input_module_size)): # 从1开始，因为第0层是输入层，不需要参数
            self.num_input_w = nn.Parameter(torch.Tensor(num_input_module_size[i], num_input_module_size[i-1]))
            self.vars.append(self.num_input_w)
            self.num_input_b = nn.Parameter(torch.Tensor(num_input_module_size[i]))
            self.vars.append(self.num_input_b)
        # 状态量输入层
        self.embedding_weight = nn.Parameter(torch.Tensor(embedding_word, embedding_dim))
        self.vars.append(self.embedding_weight)
        state_input_module_size = [embedding_dim * input_size_state] + feature_size
        for i in range(len(feature_size)):
            self.state_input_w = nn.Parameter(torch.Tensor(state_input_module_size[i+1], state_input_module_size[i]))
            self.vars.append(self.state_input_w)
            self.state_input_b = nn.Parameter(torch.Tensor(state_input_module_size[i+1]))
            self.vars.append(self.state_input_b)
        
        self.W = nn.Parameter(torch.Tensor(feature_size[-1]*2, hidden_size * 4))
        self.vars.append(self.W)
        self.U = nn.Parameter(torch.Tensor(hidden_size, hidden_size * 4))
        self.vars.append(self.U)
        self.B = nn.Parameter(torch.Tensor(hidden_size * 4))
        self.vars.append(self.B)

        # 将 h_prev 和 c_prev 设置为可训练参数
        # self.h_prev = nn.Parameter(torch.zeros(1, hidden_size), requires_grad=True)
        # self.vars.append(self.h_prev)
        # self.c_prev = nn.Parameter(torch.zeros(1, hidden_size), requires_grad=True)
        # self.vars.append(self.c_prev)

        
        self.out_w = nn.Parameter(torch.Tensor(num_cls, hidden_size))
        self.vars.append(self.out_w)
        self.out_b = nn.Parameter(torch.Tensor(num_cls))
        self.vars.append(self.out_b)

        self.reset_parameters()

    def reset_parameters(self):
        # Initialize parameters
        for p in self.parameters():
            if p.data.ndimension() >= 2:
                nn.init.xavier_uniform_(p.data)
            else:
                nn.init.zeros_(p.data)

    def forward(self, x_num, x_state, vars = None):
        # 获取序列长度
        seq_len = x_num.size(1) # 也可以用x_state.size(1)
        batch_size = x_num.size(0) # 也可以用x_state.size(0)
        cursor = 0

        if vars == None:
            vars = self.vars
    # 输入层
        for i in range(len(self.feature_size)):
            x_num = F.linear(x_num, vars[i*2], vars[i*2+1])
            x_num = F.relu(x_num)
            x_num = F.layer_norm(x_num,normalized_shape=(self.feature_size[i],))
            x_num = F.dropout(x_num, p=0.1, training=self.training)
        cursor = len(self.feature_size) * 2
    # 状态量输入层
        x_state = F.embedding(x_state, self.embedding_weight)
        cursor += 1 # 用于记录当前参数的位置，加入embedding_weight后，cursor+1
        # print(x.size())
        x_state = x_state.view(batch_size, seq_len, -1)
        # print(x.size())
        for i in range(len(self.feature_size)):
            x_state = F.linear(x_state, vars[cursor+i*2], vars[cursor+i*2+1])
            x_state = F.relu(x_state)
            x_state = F.layer_norm(x_state,normalized_shape=(self.feature_size[i],))
            x_state = F.dropout(x_state, p=0.1, training=self.training)
        cursor += len(self.feature_size) * 2
        
        x = torch.cat((x_num, x_state), dim=2) # 将模拟量和状态量拼接在一起
        # 初始化隐藏状态列表
        h_t_list, c_t_list = [], []

        # h_t = self.h_prev.expand(batch_size, -1)
        # c_t = self.c_prev.expand(batch_size, -1)
        
        # 将 h_prev 和 c_prev 设置为零
        self.h_prev = nn.Parameter(torch.zeros(1, hidden_size)).to(x.device)
        self.c_prev = nn.Parameter(torch.zeros(1, hidden_size)).to(x.device)
        h_t = self.h_prev
        c_t = self.c_prev

        # 循环更新
        for t in range(seq_len):
            gates = x[:,t,:] @ self.W + h_t @ self.U + self.B
            i_t, f_t, g_t, o_t = gates.chunk(4, 1)
            i_t = torch.sigmoid(i_t)
            f_t = torch.sigmoid(f_t)
            g_t = torch.tanh(g_t)
            o_t = torch.sigmoid(o_t)

            # Update the cell state
            c_t = f_t * c_t + i_t * g_t
            # c_t = F.dropout(f_t * c_t + i_t * g_t, p = 0.1, training=self.training) # 在生成c_t时，对c_t进行dropout
            # Update the hidden state
            h_t = o_t * torch.tanh(c_t)
                
            h_t_list.append(h_t)
            c_t_list.append(c_t)
            

        # 将所有时间步的隐藏状态和单元状态堆叠起来
        h_t = torch.stack(h_t_list)
        c_t = torch.stack(c_t_list)

        # 只取最后一个时间步的输出
        output = h_t[-1, :, :]
        # print('最后时间步：',output.size())
        # 全连接层
        output = F.linear(output, self.out_w) + self.out_b
        # 输出层无需softmax，softmax包含在CrossEntropy中
        
        return output
    
    def zero_grad(self, vars = None):
        with torch.no_grad:
            if vars == None:
                for p in self.vars:
                    if p.grad is not None:
                        p.grad.zero_()
            else:
                for p in vars:
                    if p.grad is not None:
                        p.grad.zero_()

    def parameters(self):
        return self.vars
    
if __name__ == "__main__":
    # Example usage
    batch_size = 256
    seq_len = 6
    input_size_num = 51
    input_size_state = 52
    feature_size = [128, 64]
    hidden_size = 64
    num_cls = 24
    epoch = 300
    valid_per = 0.2

    root = os.path.join(os.getcwd(),'xwdata')
    xw_data = xwdataset(root = root, twindow=seq_len, valid_per=valid_per) # 生成数据集, 用于训练，valid_per为验证集比例


    if torch.cuda.is_available(): # 判断是否有GPU
        device = torch.device('cuda:0') # 指定设备 cuda
    # elif torch.backends.mps.is_available(): # 判断是否有MPS, ARM架构的GPU
    #     device = torch.device('mps') # 指定设备 mps
    else:
        device = torch.device('cpu') # 指定设备 cpu

    model = ts_Learner(input_size_num, input_size_state, feature_size, hidden_size, num_cls).to(device)
    print(model)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.999)
    dl = DataLoader(xw_data, batch_size, shuffle=True)
    time_start = time.time()
    for iter in range(epoch):
        accuracy_sum = 0
        accuracy_sum_record = []
        loss_sum = 0
        loss_sum_record = []
        for step, (data1, data2, label) in enumerate(dl):
            # print(step, data1.size(), label.size())
            data1, data2, label = data1.to(device), data2.to(device), label.to(device)
            output = model(data1, data2)
            # print(output.size())
            # print(label.size())
            # print(output)
            # print(label)
            # break
            # loss = F.cross_entropy(output, torch.argmax(label,dim=1), reduction='mean')
            loss = nn.CrossEntropyLoss().to(device)(output, torch.max(label, dim=1)[1])
            # loss = nn.CrossEntropyLoss()(output, label)
            # for param in model.parameters():
            #     print(param, param.grad_fn)
            # torch.autograd.grad(loss, model.parameters())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            with torch.no_grad():
                loss_sum += loss.item()
                accuracy = (torch.max(output, dim=1)[1] == torch.max(label, dim=1)[1]).sum().item()
                accuracy_sum += accuracy
            # print('step:', step, 'loss:', loss.item(), 'accuracy:', accuracy)
        with torch.no_grad():
            accuracy_sum /= len(dl) * batch_size
            loss_sum /= len(dl)
        print('Epoch:', iter+1, 'loss: %.4f' % loss_sum, 'accuracy: %.4f' % accuracy_sum)
        print('learning rate: {}'.format(optimizer.param_groups[0]['lr']), flush=True)  # 查看学习率
        loss_sum_record.append(loss_sum)
        accuracy_sum_record.append(accuracy_sum)
        time_end = time.time()
        print('time cost:%.2f' % (time_end - time_start), 's')