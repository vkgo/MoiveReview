import MRdataset as mrds
import torch
import loadglove as lg
from torch.utils.data import random_split
import config
import CNNDCNN
import numpy as np

batch_size = 32
learningrate = 1e-3

# import nltk
# nltk.download('stopwords') # 这段用于下载nltk的数据

# load glove
glove_vocab_dic, emb = lg.loadGlove("../data/glove.6B.300d.txt")
emb = torch.tensor(np.array(emb))

# load dataset
total_dataset = mrds.TxtDataset(neg_loc="../raw_data/rt-polaritydata/rt-polaritydata/rt-polarity.neg", pos_loc="../raw_data/rt-polaritydata/rt-polaritydata/rt-polarity.pos", word2index=glove_vocab_dic)
train_length = int(total_dataset.__len__() * 0.7)
val_length = total_dataset.__len__() - train_length
train_dataset, val_dataset = random_split(dataset=total_dataset, lengths=[train_length, val_length])

# build dataloader
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size)

# 设置config
module_config = config.Config(len(emb), embedding_dimension=300, wordvectors=emb)

module = CNNDCNN.CNNDCNN(config=module_config)
loss_fn = torch.nn.CrossEntropyLoss()
optim = torch.optim.Adam(params=module.parameters(), lr=learningrate)

if torch.cuda.is_available():
    module.cuda()
    loss_fn.cuda()
