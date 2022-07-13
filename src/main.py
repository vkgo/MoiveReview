import MRdataset as mrds
import torch
import loadglove as lg
from torch.utils.data import random_split
import config
import CNNDCNN
import numpy as np
import time
import collate

batch_size = 32
learningrate = 1e-3
epoch = 32
useDCNN = False

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
train_dataset = train_dataset.dataset
val_dataset = val_dataset.dataset

# build dataloader
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size) # , collate_fn=collate.collate_fn
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size)

# 设置config
module_config = config.Config(len(emb), embedding_dimension=300, wordvectors=emb)

module = CNNDCNN.CNNDCNN(config=module_config)
loss_fn = torch.nn.CrossEntropyLoss()
optim = torch.optim.Adam(params=module.parameters(), lr=learningrate)

if torch.cuda.is_available():
    module.cuda()
    loss_fn.cuda()

for epoch in range(epoch):
    print("***第{}轮训练***".format(epoch + 1))
    start_time = time.time()
    counter = 0 # 打印现在是epoch里面的第几轮for
    round_loss = 0.0 # 每轮的损失
    module.train()
    for texts, labels in train_loader:
        if torch.cuda.is_available():
            texts = texts.cuda()
            labels = labels.cuda()
        counter += 1
        if useDCNN == False: # 不使用DCNN，普通的分类任务

            module_output = module(texts, False)

            loss_result = loss_fn(module_output, labels)

            optim.zero_grad()  # 梯度清零
            loss_result.backward()  # 反向传播
            optim.step()  # 生效

            round_loss += loss_result.item()

            if counter % 100 == 0:
                end_time = time.time()
                print(end_time - start_time)
                print("第{}次训练 平均损失{} 用时{}".format(counter, round_loss / 100, end_time - start_time))
                round_loss = 0.0
                start_time = time.time()
        elif useDCNN == True:
            print(111)




    # 结束一个epoch，检验正确率
    if useDCNN == False: # 不使用DCNN，普通的分类任务
        module.eval()
        testdatalength = 0
        with torch.no_grad():
            total_loss = 0.0
            counter = 0
            correct_num = 0  # 总共正确的个数

            for texts, labels in val_loader:
                testdatalength += len(labels)
                if torch.cuda.is_available():
                    texts = texts.cuda()
                    labels = labels.cuda()
                counter += 1
                module_output = module(texts, False)
                loss_result = loss_fn(module_output, labels)
                total_loss += loss_result.item()

                correct_num += (module_output.argmax(1) == labels).sum().item()

            avg_loss = total_loss / counter
            accuracy = correct_num / testdatalength
            print("##验证集每batches平均损失:{}".format(avg_loss))
            print("##验证集每batches平均正确率:{}".format(accuracy))

        # 保存模型
        # 设置隔多少epoch保存
        # epochgap = 999
        # if epoch % epochgap == 0:
        #     SavedLoc = "../SavedModel/useDCNN-" + str(useDCNN) + "/mymodule_{}.pth".format(epoch)
        #     torch.save(module, SavedLoc)
