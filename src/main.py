import MRdataset as mrds
import torch
import loadglove as lg
import nltk

batch_size = 32


# nltk.download('stopwords') # 这段用于下载nltk的数据

# load glove
glove_vocab_dic, emb = lg.loadGlove("../data/glove.6B.300d.txt")

# load dataset
dataset = mrds.TxtDataset(neg_loc="../raw_data/rt-polaritydata/rt-polaritydata/rt-polarity.neg", pos_loc="../raw_data/rt-polaritydata/rt-polaritydata/rt-polarity.pos", word2index=glove_vocab_dic)

#build dataloader
train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
