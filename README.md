# 2020语言信息处理大作业
面向数据安全治理的数据内容智能发现与分级分类代码实现（pytorch）[题目链接](https://www.datafountain.cn/competitions/471)

## 介绍
本项目用于识别样本中的敏感数据，利用远程监督技术基于小样本构建文档分类分级文本库，并与BERT模型相结合，提取文本语义特征，构建具有较强泛化能力的文档分级分类模型，判断数据所属的类别以及级别。



## 数据集
[数据集下载链接](https://www.datafountain.cn/competitions/471/datasets)

数据集包含如下数据：

1. 已标注数据labeled_data.csv：共7000篇文档，类别包含7类，分别为：财经、房产、家居、教育、科技、时尚、时政，每一类包含1000篇文档

2. 未标注数据unlabeled_data.csv：共33000篇文档

3. 分类分级测试数据test_data.csv：共20000篇文档，包含10个类别:财经、房产、家居、教育、科技、时尚、时政、游戏、娱乐、体育


每个数据样本由id、class_label（仅有标签数据）、content三个字段组成，分别代表数据id，数据所属类别以及文本内容。

文档类别与文档级别有如下对应关系：

|文档类别class_label  |文档级别rank_label|
|  ----  | ----  |
|财经、时政	|高风险|
|房产、科技	|中风险|
|教育、时尚、游戏|	低风险|
|家居、体育、娱乐|	可公开|


提交结果文件命名为“result.csv”，采用UTF-8统一编码，每个样本的预测结果包含id，class_label,rank_label三个字段。

## 环境
- python 3
- pytorch 1.1
- csv
- tqdm
- sklearn
- tensorboardX


## 预训练语言模型
本项目使用BERT预训练模型，模型下载地址如下：

bert_Chinese: [模型](https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese.tar.gz)
[词表]( https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese-vocab.txt)
[模型的网盘地址](https://pan.baidu.com/s/1qSAD5gwClq7xlgzl_4W3Pw)

下载后将对应文件（pytorch_model.bin，
bert_config.json，
vocab.txt）放在bert_pretain目录下即可。

## 使用说明
下载预训练模型，然后使用如下指令即可进行训练及测试：

```pyhton3.6 run.py```

相关模型与参数均位于models目录下的bert.py文件里。

