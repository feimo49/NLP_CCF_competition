# coding: UTF-8
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
import time
import csv
from utils_ours import get_time_dif
from pytorch_pretrained.optimization import BertAdam


def MI(outputs):
    batch_size = outputs.size(0)
    softmax_outs = nn.Softmax(dim=1)(outputs)
    avg_softmax_outs = torch.sum(softmax_outs, dim=0) / float(batch_size)
    log_avg_softmax_outs = torch.log(avg_softmax_outs)
    item1 = -torch.sum(avg_softmax_outs * log_avg_softmax_outs)
    item2 = -torch.sum(softmax_outs * torch.log(softmax_outs)) / float(batch_size)
    return item1 - item2

# 权重初始化，默认xavier
def init_network(model, method='xavier', exclude='embedding', seed=123):
    for name, w in model.named_parameters():
        if exclude not in name:
            if len(w.size()) < 2:
                continue
            if 'weight' in name:
                if method == 'xavier':
                    nn.init.xavier_normal_(w)
                elif method == 'kaiming':
                    nn.init.kaiming_normal_(w)
                else:
                    nn.init.normal_(w)
            elif 'bias' in name:
                nn.init.constant_(w, 0)
            else:
                pass

i=0

def train(config, model, train_iter,train_unlabeled_iter, dev_iter, test_iter):
    start_time = time.time()
    model.train()
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
    # optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    len_train_unlabeled = len(train_unlabeled_iter)
    len_train = len(train_iter)
    if len_train_unlabeled > len_train:
        max_len_iter = len_train_unlabeled
    else:
        max_len_iter = len_train

    optimizer = BertAdam(optimizer_grouped_parameters,
                         lr=config.learning_rate,
                         warmup=0.05,
                         t_total=max_len_iter * config.num_epochs)
    total_batch = 0   # 记录进行到多少batch
    dev_best_loss = float('inf')
    last_improve = 0  # 记录上次验证集loss下降的batch�?
    flag = False      # 记录是否很久没有效果提升
    model.train()
    for epoch in range(config.num_epochs):
        print('Epoch [{}/{}]'.format(epoch + 1, config.num_epochs))
        for i in range(max_len_iter):
            if i % len_train_unlabeled == 0:
                 train_unlabeled_iter.reset()
            if i % len_train == 0:
                train_iter.reset()

            trains_unlabeled, _ = train_unlabeled_iter.__next__()
            trains, labels = train_iter.__next__()

            outputs = model(trains)
            model.zero_grad()
            loss = F.cross_entropy(outputs, labels)

            outputs_unlabel = model(trains_unlabeled)
            loss2 = -MI(outputs_unlabel)
            total_batch = i + epoch * max_len_iter
            eta = 0.1
            total_loss=loss + eta * loss2
            total_loss.backward()
            optimizer.step()
            if i % 100 == 0:
                print('total_loss={}, loss1={}, loss2={}'.format(total_loss, loss, loss2))
                true = labels.data.cpu()
                predic = torch.max(outputs.data, 1)[1].cpu()
                train_acc = metrics.accuracy_score(true, predic)
                dev_acc, dev_loss = evaluate(config, model, dev_iter)
                if dev_loss < dev_best_loss:
                    dev_best_loss = dev_loss
                    torch.save(model.state_dict(), config.save_path)
                    improve = '*'
                    last_improve = total_batch
                else:
                    improve = ''
                msg = 'Iter: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>6.2%},  Val Loss: {3:>5.2},  Val Acc: {4:>6.2%}, {5}'
                print(msg.format(total_batch, total_loss.item(), train_acc, dev_loss, dev_acc, improve))
                if total_batch - last_improve > config.require_improvement:
                # 验证集loss超过1000batch没下降，结束训练
                    print("No optimization for a long time, auto-stopping...")
                    flag = True
                    break
        if flag:
            break
    test(config, model, dev_iter, test_iter)


def test(config, model, val_iter, test_iter):
    # test
    model.load_state_dict(torch.load(config.save_path))
    model.eval()
    start_time = time.time()

    test_acc, test_loss, test_report, test_confusion = evaluate(config, model, val_iter, test=True)
    msg = 'Test Loss: {0:>5.2},  Val Acc: {1:>6.2%}'
    print(msg.format(test_loss, test_acc))
    print("Precision, Recall and F1-Score...")
    print(test_report)
    print("Confusion Matrix...")
    print(test_confusion)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

    evaluate_for_test(config, model, test_iter, test=True)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)


def evaluate(config, model, data_iter, test=False):
    model.eval()
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    with torch.no_grad():
        #for i, (texts, labels) in enumerate(data_iter):
        data_iter.reset()
        for i in range(len(data_iter)):
            texts, labels = data_iter.__next__()
            outputs = model(texts)
            loss = F.cross_entropy(outputs, labels)
            loss_total += loss
            labels = labels.data.cpu().numpy()
            predic = torch.max(outputs.data, 1)[1].cpu().numpy()
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predic)

    acc = metrics.accuracy_score(labels_all, predict_all)
    if test:
        report = metrics.classification_report(labels_all, predict_all, target_names=config.class_list, digits=4)
        confusion = metrics.confusion_matrix(labels_all, predict_all)
        return acc, loss_total / len(data_iter), report, confusion
    return acc, loss_total / len(data_iter)



def evaluate_for_test(config, model, data_iter, test=False):
    class_dic = {0: '财经', 1: '时政', 2: '房产', 3: '科技', 4: '教育', 5: '时尚', 6: '游戏', 7: '家居', 8: '体育', 9: '娱乐'}
    rank_dic = {0: '高风险', 1: '中风险', 2: '低风险', 3: '可公开'}

    model.eval()
    predict_all = np.array([], dtype=int)
    with torch.no_grad():
        data_iter.reset()
        for i in range(len(data_iter)):
            texts, _ = data_iter.__next__()
            outputs = model(texts)
            predic = torch.max(outputs.data, 1)[1].cpu().numpy()
            predict_all = np.append(predict_all, predic)

        csvfile = open('result.csv', mode='w')
    writer = csv.writer(csvfile)
    writer.writerow(['id', 'class_label', 'rank_label'])
    for i in range(len(predict_all)):
        predict_class_label = predict_all[i]
        if predict_class_label == 0 or predict_class_label == 1:
            rank_label = 0
        elif predict_class_label == 2 or predict_class_label == 3:
            rank_label = 1
        elif predict_class_label == 4 or predict_class_label == 5 or predict_class_label == 6:
            rank_label = 2
        elif predict_class_label == 7 or predict_class_label == 8 or predict_class_label == 9:
            rank_label = 3
        print(i, class_dic[predict_class_label], rank_dic[rank_label])
        data = [str(i), class_dic[predict_class_label], rank_dic[rank_label]]
        writer.writerow(data)
    csvfile.close()