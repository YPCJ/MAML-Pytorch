import os
import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms
import numpy as np
import collections
from PIL import Image
import csv
import random


class MiniImagenet(Dataset):
    """
    put mini-imagenet files as :
    root :
        |- images/*.jpg includes all imgeas
        |- train.csv
        |- test.csv
        |- val.csv
    NOTICE: meta-learning is different from general supervised learning, especially the concept of batch and set.
    batch: contains several sets
    sets: conains n_way * k_shot for meta-train set, n_way * n_query for meta-test set.
    """

    def __init__(self, root, mode, batchsz, n_way, k_shot, k_query, resize, startidx=0):
        """

        :param root: root path of mini-imagenet
        :param mode: train, val or test
        :param batchsz: batch size of sets, not batch of imgs
        :param n_way:
        :param k_shot:
        :param k_query: num of qeruy imgs per class
        :param resize: resize to
        :param startidx: start to index label from startidx
        """

        self.batchsz = batchsz  # batch of set, not batch of imgs
        self.n_way = n_way  # n-way
        self.k_shot = k_shot  # k-shot
        self.k_query = k_query  # for evaluation
        self.setsz = self.n_way * self.k_shot  # num of samples per set # 每个任务的支持集的样本数，等于n_way * k_shot
        self.querysz = self.n_way * self.k_query  # number of samples per set for evaluation # 每个任务的查询集的样本数，等于n_way * k_query
        self.resize = resize  # resize to
        self.startidx = startidx  # index label not from 0, but from startidx
        print('shuffle DB :%s, b:%d, %d-way, %d-shot, %d-query, resize:%d' % (
        mode, batchsz, n_way, k_shot, k_query, resize))

        if mode == 'train': # 训练时进行数据增强
            self.transform = transforms.Compose([lambda x: Image.open(x).convert('RGB'), # 将图片转换为RGB格式
                                                 transforms.Resize((self.resize, self.resize)), # 将图片resize为resize*resize
                                                #  transforms.RandomHorizontalFlip(), # 随机水平翻转，原本1.0版本的有这一步，2.0版本给注释掉了
                                                #  transforms.RandomRotation(5), # 随机旋转5度，原本1.0版本的有这一步，2.0版本给注释掉了
                                                 transforms.ToTensor(), # 将图片转换为tensor
                                                 transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)) # 对图片进行归一化，均值和方差来自ImageNet
                                                 ])
        else: # 测试时不进行数据增强
            self.transform = transforms.Compose([lambda x: Image.open(x).convert('RGB'),
                                                 transforms.Resize((self.resize, self.resize)),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                                                 ])

        self.path = os.path.join(root, 'images')  # image path # 图片路径，os.path.join()函数用于拼接路径
        csvdata = self.loadCSV(os.path.join(root, mode + '.csv'))  # csv path # csv文件路径，self.loadCSV()函数用于读取csv文件，这个函数在下面定义
        self.data = [] # 用于存储所有图片的路径
        self.img2label = {} # 用于存储图片的标签
        for i, (k, v) in enumerate(csvdata.items()): # 遍历csv文件中的每一行
            self.data.append(v)  # [[img1, img2, ...], [img111, ...]] # 将图片路径添加到data中，每个类别对应一个列表，列表中存储该类别的所有图片的路径，data中存储所有类别的图片的路径
            self.img2label[k] = i + self.startidx  # {"img_name[:9]":label} # 将图片的标签添加到img2label中，img_name[:9]表示图片的前9个字符，这9个字符作为图片的标签
        self.cls_num = len(self.data) # 类别数，即data的长度

        self.create_batch(self.batchsz) # 创建batch，即创建batchsz个任务，每个任务包括支持集和查询集

    def loadCSV(self, csvf): # 读取csv文件
        """
        return a dict saving the information of csv
        :param splitFile: csv file name
        :return: {label:[file1, file2 ...]}
        """
        dictLabels = {} # 用于存储csv文件中的信息，{label:[file1, file2 ...]}，是一个字典，key为标签，value为图片的文件名
        with open(csvf) as csvfile:
            csvreader = csv.reader(csvfile, delimiter=',')
            next(csvreader, None)  # skip (filename, label) # 跳过第一行
            for i, row in enumerate(csvreader): # 遍历csv文件中的每一行
                filename = row[0] # 图片的文件名
                label = row[1] # 图片的标签
                # append filename to current label
                if label in dictLabels.keys(): # 如果标签已经在dictLabels中
                    dictLabels[label].append(filename) # 将图片的文件名添加到dictLabels中，即添加到对应的标签中
                else:
                    dictLabels[label] = [filename] # 如果标签不在dictLabels中，将图片的文件名添加到dictLabels中，即添加到对应的标签中
        return dictLabels # 返回dictLabels，即返回一个字典，key为标签，value为图片的文件名，{label1:[file11, file12 ...], label2:[file21, file22 ...] ...}

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

    def __getitem__(self, index): # 获取一个任务的支持集和查询集，index表示任务的索引
        """
        index means index of sets, 0<= index <= batchsz-1
        :param index:
        :return:
        """
        # [setsz, 3, resize, resize]
        support_x = torch.FloatTensor(self.setsz, 3, self.resize, self.resize) # 用于存储一个任务的支持集,setsz表示一个任务的支持集的样本数
        # [setsz]
        support_y = np.zeros((self.setsz), dtype=np.int) # 用于存储一个任务的支持集的标签
        # [querysz, 3, resize, resize]
        query_x = torch.FloatTensor(self.querysz, 3, self.resize, self.resize) # 用于存储一个任务的查询集,querysz表示一个任务的查询集的样本数
        # [querysz]
        query_y = np.zeros((self.querysz), dtype=np.int) # 用于存储一个任务的查询集的标签

        flatten_support_x = [os.path.join(self.path, item)
                             for sublist in self.support_x_batch[index] for item in sublist] # 将一个任务的支持集的图片的文件名转换为图片的路径，flatten_support_x是一个列表，列表中存储一个任务的所有支持集的图片的路径
        support_y = np.array(
            [self.img2label[item[:9]]  # filename:n0153282900000005.jpg, the first 9 characters treated as label
             for sublist in self.support_x_batch[index] for item in sublist]).astype(np.int32) # 前9个字符就是图片的标签，support_y是一个列表，列表中存储一个任务的所有支持集的图片的标签

        flatten_query_x = [os.path.join(self.path, item) # 将一个任务的查询集的图片的文件名转换为图片的路径，flatten_query_x是一个列表，列表中存储一个任务的所有查询集的图片的路径
                           for sublist in self.query_x_batch[index] for item in sublist]
        query_y = np.array([self.img2label[item[:9]]
                            for sublist in self.query_x_batch[index] for item in sublist]).astype(np.int32) # 前9个字符就是图片的标签，query_y是一个列表，列表中存储一个任务的所有查询集的图片的标签

        # print('global:', support_y, query_y)
        # support_y: [setsz]
        # query_y: [querysz]
        # unique: [n-way], sorted
        unique = np.unique(support_y) # 获取支持集的标签中的所有类别，unique是一个列表，列表中存储支持集的标签中的所有类别
        random.shuffle(unique) # 打乱类别的顺序，这步操作的目的是为了将支持集和查询集的标签转换为相对标签，即将原本的标签转换为0到n_way-1的标签，以形成一个独立的任务
        # 虽然unique的顺序乱了，但是unique中的元素没有变，所以后面的操作不会受到影响
        # relative means the label ranges from 0 to n-way
        support_y_relative = np.zeros(self.setsz) # 用于存储一个任务的支持集的相对标签
        query_y_relative = np.zeros(self.querysz) # 用于存储一个任务的查询集的相对标签
        for idx, l in enumerate(unique):
            support_y_relative[support_y == l] = idx # 将支持集的标签转换为相对标签，即将原本的标签转换为0到n_way-1的标签, support_y == l表示找到支持集中标签为l的索引，将这些索引的值赋值为idx
            query_y_relative[query_y == l] = idx # 将查询集的标签转换为相对标签，即将原本的标签转换为0到n_way-1的标签, query_y == l表示找到查询集中标签为l的索引，将这些索引的值赋值为idx

        # print('relative:', support_y_relative, query_y_relative)

        for i, path in enumerate(flatten_support_x): # 遍历一个任务的所有支持集的图片的路径
            support_x[i] = self.transform(path) # 将图片转换为tensor，存储到support_x中，support_x是定义好的tensor，用于存储一个任务的所有支持集的图片数据

        for i, path in enumerate(flatten_query_x): # 遍历一个任务的所有查询集的图片的路径
            query_x[i] = self.transform(path) # 将图片转换为tensor，存储到query_x中，query_x是定义好的tensor，用于存储一个任务的所有查询集的图片数据
        # print(support_set_y)
        # return support_x, torch.LongTensor(support_y), query_x, torch.LongTensor(query_y)

        return support_x, torch.LongTensor(support_y_relative), query_x, torch.LongTensor(query_y_relative)
        # 返回一个任务的支持集和查询集，支持集和查询集的标签是相对标签，即将原本的标签转换为0到n_way-1的标签

    def __len__(self):
        # as we have built up to batchsz of sets, you can sample some small batch size of sets.
        return self.batchsz # 返回batchsz，即任务的个数


if __name__ == '__main__':
    # the following episode is to view one set of images via tensorboard.
    from torchvision.utils import make_grid
    from matplotlib import pyplot as plt
    from tensorboardX import SummaryWriter
    import time

    plt.ion() # 开启交互模式

    tb = SummaryWriter('runs', 'miniimagenet') # 实例化SummaryWriter类，用于可视化
    mini = MiniImagenet('./miniimagenet/', mode='train', n_way=5, k_shot=1, k_query=1, batchsz=1000, resize=168)

    for i, set_ in enumerate(mini):
        # support_x: [k_shot*n_way, 3, 84, 84]
        support_x, support_y, query_x, query_y = set_ # 获取一个任务的支持集和查询集
        support_x = make_grid(support_x, nrow=support_y.size(0)) # 将支持集的图片拼接成网格，nrow表示每行的图片数
        query_x = make_grid(query_x, nrow=support_y.size(0)) # 将查询集的图片拼接成网格，nrow表示每行的图片数

        plt.figure(1)
        plt.imshow(support_x.transpose(2, 0).transpose(1, 0).numpy()) # 将图片转换为numpy格式，然后转置，最后显示图片,transpose(2, 0)表示将第三个维度放到第一个维度，第一个维度放到第三个维度，第二个维度不变
        # plt.imshow(support_x.numpy()) # 将图片转换为numpy格式，然后显示图片
        plt.pause(0.5)
        plt.figure(2)
        plt.imshow(query_x.transpose(2, 0).transpose(1, 0).numpy()) # 将图片转换为numpy格式，然后转置，最后显示图片
        # plt.imshow(query_x.numpy()) # 将图片转换为numpy格式，然后显示图片
        plt.pause(0.5)
        

        tb.add_image('support_x_'+str(i), support_x) # 将支持集的图片添加到tensorboard中
        tb.add_image('query_x_'+str(i), query_x) # 将查询集的图片添加到tensorboard中

        time.sleep(20) # 等待20秒

    tb.close()
