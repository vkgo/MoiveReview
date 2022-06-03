import pandas as pd
import os
from torch.utils.data.dataset import Dataset




class CSVDataset(Dataset):
    def __init__(self, dirloc, word_to_index):
        self.Text = pd.Series()
        self.Target = pd.Series()
        # neg
        neg_path = os.path.join(dirloc, 'neg/')
        for dirpath, dirnames, filenames in os.walk(neg_path):
            for filename in filenames:
                if filename[-4:] == '.txt':
                    fileloc = os.path.join(dirpath, filename)  # 文件的绝对路径
                    with open(fileloc, 'r', encoding='utf-8') as f:
                        self.Text.append(f.readlines())
                        self.Target.append(0.)

        # pos
        pos_path = os.path.join(dirloc, 'pos/')
        for dirpath, dirnames, filenames in os.walk(pos_path):
            for filename in filenames:
                if filename[-4:] == '.txt':
                    fileloc = os.path.join(dirpath, filename)  # 文件的绝对路径
                    with open(fileloc, 'r', encoding='utf-8') as f:
                        self.Text.append(f.readlines())
                        self.Target.append(0.)


