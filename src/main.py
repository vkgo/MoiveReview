import MRdataset as mrds
import torch
import loadglove as lg
from torch.utils.data import random_split
import config
import CNNDCNN
import numpy as np
import time
import matrixloss as mloss

batch_size = 32
learningrate = 1e-3
CNNDCNNepoch = 32
FCepoch = 32
useDCNN = True
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

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
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True) # , collate_fn=collate.collate_fn
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

# 设置config
module_config = config.Config(len(emb), embedding_dimension=300, wordvectors=emb)

module = CNNDCNN.CNNDCNN(config=module_config).to(device)
fcmodule = CNNDCNN.fclayer().to(device)
loss_fn = torch.nn.CrossEntropyLoss().to(device)
loss_matrix = mloss.MatrixLoss().to(device)
optim = torch.optim.Adam(params=module.parameters(), lr=learningrate)
fcoptim = torch.optim.Adam(params=fcmodule.parameters(), lr=learningrate)


for epoch in range(CNNDCNNepoch):
    print("***第{}轮训练***".format(epoch + 1))
    start_time = time.time()
    counter = 0 # 打印现在是epoch里面的第几轮for
    round_loss = 0.0 # 每轮的损失
    module.train()
    for texts, labels in train_loader:
        texts = texts.to(device)
        labels = labels.to(device)
        counter += 1
        ori_x, module_output = module(texts, True)
        loss_result = loss_matrix(module_output, ori_x)

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

for epoch in range(FCepoch):
    print("***第{}轮训练***".format(epoch + 1))
    start_time = time.time()
    counter = 0  # 打印现在是epoch里面的第几轮for
    round_loss = 0.0  # 每轮的损失
    module.eval()
    fcmodule.train()
    for texts, labels in train_loader:
        texts = texts.to(device)
        labels = labels.to(device)
        counter += 1
        middle_x = module(texts, False)
        module_output = fcmodule(middle_x)

        loss_result = loss_fn(module_output, labels)


        fcoptim.zero_grad()  # 梯度清零
        loss_result.backward()  # 反向传播
        fcoptim.step()  # 生效

        round_loss += loss_result.item()

        if counter % 100 == 0:
            end_time = time.time()
            print(end_time - start_time)
            print("第{}次训练 平均损失{} 用时{}".format(counter, round_loss / 100, end_time - start_time))
            round_loss = 0.0
            start_time = time.time()
    # 结束一个epoch，检验正确率
    fcmodule.eval()
    testdatalength = 0
    with torch.no_grad():
        total_loss = 0.0
        counter = 0
        correct_num = 0  # 总共正确的个数

        for texts, labels in val_loader:
            testdatalength += len(labels)
            texts = texts.to(device)
            labels = labels.to(device)
            counter += 1
            middle_x = module(texts, False)
            module_output = fcmodule(middle_x)
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
