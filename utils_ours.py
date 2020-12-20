# coding: UTF-8
import torch
from tqdm import tqdm
import time
import pandas as pd
import random

from datetime import timedelta

PAD, CLS = '[PAD]', '[CLS]'  # padding符号, bert中综合信息符号

class_dict = {'财经':0, '时政':1, '房产':2, '科技':3, '教育':4, '时尚':5, '游戏':6, '家居':7, '体育':8,  '娱乐':9}

def build_dataset(config):

    def load_dataset(path, labeled=True, pad_size=100):
        data = pd.read_csv(path, sep=',', encoding='utf-8')
        num_samples = len(data['content'])
        print('num_samples = {}'.format(num_samples))
        contents = []
        for i in range(num_samples):
            line = data['content'][i]
            content = line.strip()  # 去掉首尾空白符
            token = config.tokenizer.tokenize(content)                 # 用bert内置的tokenizer进行分字操作（character level， 例如，‘我’，‘们’）
            token = [CLS] + token   # 头部加入[CLS]
            seq_len = len(token)    # 文本实际长度
            mask = []               # 区分填充部分和非填充部分
            token_ids = config.tokenizer.convert_tokens_to_ids(token)  # 将token转换为索引（基于下载的词表文件）

            if pad_size:            # 长截短填操作，将长度统一为pad_size
                if len(token) < pad_size:
                   mask = [1] * len(token_ids) + [0] * (pad_size - len(token))  # 填充部分对应0，非填充部分对应1
                   token_ids += ([0] * (pad_size - len(token)))                 # seq_len为文本实际长度
                else:
                   mask = [1] * pad_size
                   token_ids = token_ids[:pad_size]  # 截断操作
                   seq_len = pad_size                # seq_len设置为pad_size
            if labeled:
                label = class_dict[data['class_label'][i]]
                contents.append((token_ids, int(label), seq_len, mask))  # [([...], label, seq_len, [...])]
            else:
                contents.append((token_ids, -1 , seq_len, mask))  # [([...], -1, seq_len, [...])]
        return contents
    train = load_dataset(config.train_path, labeled=True, pad_size=config.pad_size)
    train_unlabeled = load_dataset(config.train_unlabeled_path, labeled=True, pad_size=config.pad_size)
    test = load_dataset(config.test_path, labeled=False, pad_size=config.pad_size)
    return train,train_unlabeled, test

class DatasetIterater(object):
    def __init__(self, batches, batch_size, device, shuffle):
        self.batch_size = batch_size
        self.batches = batches
        if shuffle:
            random.shuffle(self.batches)
        self.shuffle = shuffle
        self.n_batches = len(batches) // batch_size
        self.residue = False  # 记录batch数量是否为整数
        if len(batches) % self.n_batches != 0:
            self.residue = True
        self.index = 0
        self.device = device


    def _to_tensor(self, datas):
        x = torch.LongTensor([_[0] for _ in datas]).to(self.device)        # x对应token_ids
        y = torch.LongTensor([_[1] for _ in datas]).to(self.device)        # y对应label
        seq_len = torch.LongTensor([_[2] for _ in datas]).to(self.device)  # pad前的长度(超过pad_size的设为pad_size)
        mask = torch.LongTensor([_[3] for _ in datas]).to(self.device)     # mask标记是否是填充部分
        return (x, seq_len, mask), y

    def __next__(self):
        if self.residue and self.index == self.n_batches:
            batches = self.batches[self.index * self.batch_size: len(self.batches)]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

        elif self.index >= self.n_batches:
            self.index = 0
            raise StopIteration

        else:
            batches = self.batches[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

    def __iter__(self):
        return self

    def reset(self):
        self.index = 0
        if self.shuffle:
            random.shuffle(self.batches)

    def __len__(self):
        if self.residue:
            return self.n_batches + 1
        else:
            return self.n_batches


def build_iterator(dataset, config, shuffle):
    iter = DatasetIterater(dataset, config.batch_size, config.device, shuffle)
    return iter


def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))
