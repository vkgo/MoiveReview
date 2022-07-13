import torch
from torch.utils.data.dataset import Dataset
import loadglove as lg
import numpy as np

class TxtDataset(Dataset):
    # 此Dataset适合读入数据都在一个txt，按行划分，同一个txt内的标签是txt文件名
    def __init__(self, neg_loc, pos_loc, word2index):
        self.text = torch.Tensor()
        # self.text = []
        self.target = []



        # neg
        with open(neg_loc, "r", encoding="utf-8") as f:
            lines = f.readlines()
            for line in lines:
                indexes = lg.sentence2index(line, dictionary=word2index, minlength=60, maxlength=60)
                self.text = torch.cat((self.text, torch.as_tensor([indexes])))
                # self.text = torch.cat((self.text, torch.tensor(indexes, dtype=torch.long)))
                # self.text.append(torch.as_tensor(indexes))
                # self.target.append(0)
                self.target.append(torch.tensor(0, dtype=torch.long))

        # pos
        with open(pos_loc, "r", encoding="utf-8") as f:
            lines = f.readlines()
            for line in lines:
                indexes = lg.sentence2index(line, dictionary=word2index, minlength=60, maxlength=60)
                self.text = torch.cat((self.text, torch.as_tensor([indexes])))
                # self.text = torch.cat((self.text, torch.tensor(indexes, dtype=torch.long)))
                # self.text.append(indexes)
                # self.target.append(1)
                self.target.append(torch.tensor(1, dtype=torch.long))


        # self.text = self.text.resize(sizes=[self.text.size() / 60, 60])
        # self.text = torch.cat((i for i in self.text), dim=0)
        # self.target = torch.cat((i for i in self.target), dim=0)
        # self.text = torch.as_tensor(self.text)
        # self.target = torch.as_tensor(self.target)

    def __getitem__(self, index):
        return self.text[index], self.target[index]

    def __len__(self):
        return len(self.text)