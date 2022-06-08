from torch.utils.data.dataset import Dataset
import loadglove as nlg

class TxtDataset(Dataset):
    # 此Dataset适合读入数据都在一个txt，按行划分，同一个txt内的标签是txt文件名
    def __init__(self, neg_loc, pos_loc, word_to_index):
        self.text = []
        self.target = []

        # neg
        with open(neg_loc, "r", encoding="utf-8") as f:
            lines = f.readlines()
            for line in lines:
                indexes = nlg.sentence2index(line)
                self.text.append(indexes)
                self.target.append(0)
        # pos
        with open(pos_loc, "r", encoding="utf-8") as f:
            lines = f.readlines()
            for line in lines:
                indexes = nlg.sentence2index(line)
                self.text.append(indexes)
                self.target.append(1)

    def __getitem__(self, index):
        return self.text[index], self.target[index]

    def __len__(self):
        return len(self.text)