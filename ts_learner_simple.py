import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from xwdataset import xwdataset
import os
from tqdm import tqdm
import time

class ts_Learner_sp(nn.Module):
## 时序学习器，用于时序数据的分类模型，包含模拟量输入层，状态量嵌入层和输入层，LSTM层，输出层
    def __init__(self, input_size_num, input_size_state, feature_size, hidden_size, num_cls, embedding_word = 4, embedding_dim = 4):
        super(ts_Learner_sp, self).__init__()
        self.input_size_num = input_size_num
        self.input_size_state = input_size_state
        self.feature_size = feature_size
        self.embedding_dim = embedding_dim
        self.embedding_word = embedding_word
        self.hidden_size = hidden_size
        self.num_cls = num_cls
        self.training = True
        self.no_input_layer = False

        self.vars = nn.ParameterList()
        
        # 模拟量输入层
        num_input_module_size = [input_size_num] + feature_size
        self.no_input_layer = False
        self.input_layer_num = nn.ModuleList(nn.Linear(num_input_module_size[i], num_input_module_size[i+1]) for i in range(len(feature_size)))
        self.activation_num = nn.ModuleList(nn.ReLU() for i in range(len(feature_size)))
        self.dropout_num = nn.ModuleList(nn.Dropout(p=0.1) for i in range(len(feature_size)))

        # 状态量输入层
        self.embedding_layer = nn.Embedding(embedding_word, embedding_dim)
        state_input_module_size = [embedding_dim * input_size_state] + feature_size
        self.input_layer_state = nn.ModuleList(nn.Linear(state_input_module_size[i], state_input_module_size[i+1]) for i in range(len(feature_size)))
        self.activation_state = nn.ModuleList(nn.ReLU() for i in range(len(feature_size)))
        self.dropout_state = nn.ModuleList(nn.Dropout(p=0.1) for i in range(len(feature_size)))

        self.lstm = nn.LSTM(input_size = num_input_module_size[-1] + state_input_module_size[-1], hidden_size = hidden_size, batch_first = True, dropout = 0.1)

        
        self.out = nn.Linear(hidden_size, num_cls)


    def forward(self, x_num, x_state, vars = None):
        # 获取序列长度
        seq_len = x_num.size(1) # 也可以用x_state.size(1)
        batch_size = x_num.size(0) # 也可以用x_state.size(0)
        
        for linear,activation,dropout in zip(self.input_layer_num, self.activation_num, self.dropout_num):
            x_num = linear(x_num)
            x_num = activation(x_num)
            x_num = dropout(x_num)
        x_state = self.embedding_layer(x_state)
        x_state = x_state.view(batch_size, seq_len, -1) # 将嵌入层的输出展平
        for linear,activation,dropout in zip(self.input_layer_state, self.activation_state, self.dropout_state):
            x_state = linear(x_state)
            x_state = activation(x_state)
            x_state = dropout(x_state)
        x = torch.cat((x_num, x_state), dim=2)
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        output = self.out(x)
        return output
    
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

    model = ts_Learner_sp(input_size_num, input_size_state, feature_size, hidden_size, num_cls).to(device)
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
            accuracy = (torch.max(output, dim=1)[1] == torch.max(label, dim=1)[1]).sum().item()
            # for param in model.parameters():
            #     print(param, param.grad_fn)
            # torch.autograd.grad(loss, model.parameters())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            with torch.no_grad():
                loss_sum += loss.item()
            # print('step:', step, 'loss:', loss.item(), 'accuracy:', accuracy)
            accuracy_sum += accuracy
        accuracy_sum /= len(dl) * batch_size
        loss_sum /= len(dl)
        print('Epoch:', iter+1, 'loss: %.4f' % loss_sum, 'accuracy: %.4f' % accuracy_sum)
        print('learning rate: {}'.format(optimizer.param_groups[0]['lr']), flush=True)  # 查看学习率
        loss_sum_record.append(loss_sum)
        accuracy_sum_record.append(accuracy_sum)
        time_end = time.time()
        print('time cost:%.2f' % (time_end - time_start), 's')