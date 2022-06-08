# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import os
import pickle as cPickle
import gensim
import math
import random

def glove_w2v():

    # 数据有问题，utf-8不能解码某些位置
    with open('../raw_data/rt-polaritydata/rt-polarity.pos', 'r', encoding='utf-8', errors='ignore') as f:
        pos_lines = f.readlines()
    with open('../raw_data/rt-polaritydata/rt-polarity.neg', 'r', encoding='utf-8', errors='ignore') as f:
        neg_lines = f.readlines()

    pos_data = []
    for pos_l in pos_lines:
        pos_d, _ = Process_sent(pos_l)
        pos_data.append(pos_d)
    
    neg_data = []
    for neg_l in neg_lines:
        neg_d, _ = Process_sent(neg_l)
        neg_data.append(neg_d)

    datas = pos_data + neg_data
    labels = np.concatenate((np.ones(5331), np.zeros(5331)), axis=0) #不了解axis

    dataset_list = []
    for data, label in zip(datas, labels):
        dataset_list.append([data, label])
    random.shuffle(dataset_list)

    total_num = len(dataset_list)
    train_num = int(0.9 * total_num)
    test_num = total_num - train_num

    train_list = []
    test_list = []

    L = [x for x in range(total_num)]
    lis = []
    p = 0
    for i in range(10):
        if i == 9:
            lis.append(L[p:])
        else:
            lis.append(L[p:p+int(total_num/10)])
        p = p + int(total_num/10)

    L = []
    for i in range(10):
        l = []
        l.extend(lis[i])
        for j in range(10):
            if j != i:
                l.extend(lis[j])
        L.append(l)

    for n in range(10):
        print(n)
        num = 0
        with open('MR.test.all.'+str(n), 'w', encoding='utf-8') as f:
            for m in range(len(lis[n])):
                f.write(str(int(dataset_list[L[n][num]][1])))
                f.write(' ')
                f.write(' '.join(dataset_list[L[n][num]][0]))
                num += 1
        with open('MR.train.all.'+str(n), 'w', encoding='utf-8') as f:
            for m in range(total_num - len(lis[n])):
                f.write(str(int(dataset_list[L[n][num]][1])))
                f.write(' ')
                f.write(' '.join(dataset_list[L[n][num]][0]))
                num += 1

def Process_sent(sent):
    
    sent = sent.replace('.', '')
    sent = sent.replace('"', '')
    sent = sent.replace(',', '')
    sent = sent.replace('(', '')
    sent = sent.replace(')', '')
    sent = sent.replace('(', '')
    sent = sent.replace('!', '')
    sent = sent.replace('?', '')
    
    sent = sent.replace('\'','')
    sent = sent.replace('...', '')
    sent = sent.replace("\/", ' ')
    
    sent_list = sent.split(' ')
    while "" in sent_list:
        sent_list.remove("")
    length = len(sent_list)
    return sent_list, length 

if __name__ == '__main__':
    glove_w2v()