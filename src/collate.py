import torch
from torch.nn.utils.rnn import pad_sequence
def collate_fn(batch_data):
    """
    自定义 batch 内各个数据条目的组织方式
    :param data: 元组，第一个元素：句子序列数据，第二个元素：长度 第2维：句子标签
    :return: 填充后的句子列表、实际长度的列表、以及label列表
    """
    # batch_data 为一个batch的数据组成的列表，data中某一元素的形式如下
    # (tensor([1, 2, 3, 5]), 4, 0)
    # 后续将填充好的序列数据输入到RNN模型时需要使用pack_padded_sequence函数
    # pack_padded_sequence函数要求要按照序列的长度倒序排列
    batch_data.sort(key=lambda xi: len(xi[0]), reverse=True)
    sent_seq = [xi[0] for xi in batch_data]
    label = [xi[1] for xi in batch_data]
    # padded_sent_seq = pad_sequence(sent_seq, batch_first=True, padding_value=0) # 等长处理
    return sent_seq, torch.tensor(label, dtype=torch.long)
