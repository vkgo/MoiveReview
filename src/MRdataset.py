import torch
from torch.utils.data.dataset import Dataset
import loadglove as lg
import numpy as np

class TxtDataset(Dataset):
    # 此Dataset适合读入数据都在一个txt，按行划分，同一个txt内的标签是txt文件名
    def __init__(self, neg_loc, pos_loc, word2index):
        self.text = torch.Tensor()
        self.target = []

        with open(neg_loc, "r", encoding="utf-8") as f_neg:
            with open(pos_loc, "r", encoding="utf-8") as f_pos:
                lines_neg = f_neg.readlines()
                lines_pos = f_pos.readlines()
                for line in lines_neg:
                    indexes = lg.sentence2index(line, dictionary=word2index, minlength=57, maxlength=57)
                    self.text = torch.cat((self.text, torch.as_tensor([indexes])))
                    self.target.append(torch.tensor(0, dtype=torch.long))
                for line in lines_pos:
                    indexes = lg.sentence2index(line, dictionary=word2index, minlength=57, maxlength=57)
                    self.text = torch.cat((self.text, torch.as_tensor([indexes])))
                    self.target.append(torch.tensor(1, dtype=torch.long))



    def __getitem__(self, index):
        return self.text[index], self.target[index]

    def __len__(self):
        return len(self.text)