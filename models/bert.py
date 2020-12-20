# coding: UTF-8
import torch
import torch.nn as nn
import os
import os.path as osp
# from pytorch_pretrained_bert import BertModel, BertTokenizer
from pytorch_pretrained import BertModel, BertTokenizer


class Config(object):

    """配置参数"""
    def __init__(self, dataset):
        self.model_name = 'bert'
        self.train_path = dataset + '/data/labeled_dataA_new.csv'  # 训练路径
        self.train_unlabeled_path = dataset + '/data/unlabeled_to_labeled.csv'
        self.test_path = dataset + '/data/test_data.csv'  # 测试路径
        self.class_list = [x.strip() for x in open(dataset + '/data/class_ours.txt').readlines()]     # 类别名单
        self.save_path = dataset + '/saved_dict/' + self.model_name + '.ckpt'        # 模型训练结果
        if not osp.exists(dataset + '/saved_dict/'):
            os.makedirs(dataset + '/saved_dict/')
        self.device = torch.device("cuda:5")
        self.require_improvement = 1000                                 # 若超1000batch效果还没提升，则提前结束训练
        self.num_classes = len(self.class_list)                         # 类别数
        self.num_epochs = 3                                             # epoch数
        self.batch_size = 32                                            # mini-batch大小
        self.pad_size = 100                                             # 每句话处理成的长度（短填长切)
        self.learning_rate = 1e-6                                       # 学习率
        self.bert_path = './bert_pretrain'
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)
        self.hidden_size = 768

        print('num_epochs = {}\nbatch_size = {}\npad_size = {}\nlearning_rate={}'.format(self.num_epochs, self.batch_size, self.pad_size, self.learning_rate))


class Model(nn.Module):

    def __init__(self, config):
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained(config.bert_path)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.fc = nn.Linear(config.hidden_size, config.num_classes)

    def forward(self, x):
        context = x[0]  # 输入的句子
        mask = x[2]  # 对padding部分进行mask，和句子一个size，padding部分用0表示，如：[1, 1, 1, 1, 0, 0]
        _, pooled = self.bert(context, attention_mask=mask, output_all_encoded_layers=False)
        out = self.fc(pooled)
        return out
