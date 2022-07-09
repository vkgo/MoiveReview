from torch import nn
import torch

class CNNDCNN(nn.Module):
    # 这是一个3层CNN-DCNN AutoEncoder模型
    def __init__(self, config):
        # config需包含
        # 1. vocabulary_length 2. embedding_dimension 3. wordvectors
        super(CNNDCNN, self).__init__()

        self.embedding = nn.Embedding(config.vocabulary_length, config.embedding_dimension)  # 定义embedding层
        self.embedding = self.embedding.from_pretrained(config.wordvectors)  # 加载词向量


        # CNN Layers
        # self.encoders = nn.Sequential(
        #     nn.Conv2d(in_channels=1, out_channels=300, kernel_size=[config.embedding_dimension, 5], stride=2, bias=True),
        #     nn.Conv2d(in_channels=300, out_channels=600, kernel_size=[1, 5], stride=2, bias=True),
        #     nn.Conv2d(in_channels=600, out_channels=500, kernel_size=[1, 12], stride=1, bias=True)
        # )
        self.encoder1 = nn.Conv2d(in_channels=1, out_channels=300, kernel_size=[config.embedding_dimension, 5], stride=2, bias=True)
        self.encoder2 = nn.Conv2d(in_channels=300, out_channels=600, kernel_size=[1, 5], stride=2, bias=True)
        self.encoder3 = nn.Conv2d(in_channels=600, out_channels=500, kernel_size=[1, 12], stride=1, bias=True)

        # DCNN Layers
        # self.decoders = nn.Sequential(
        #     nn.ConvTranspose2d(in_channels=500, out_channels=600, kernel_size=[1, 12], stride=1, bias=True),
        #     nn.ConvTranspose2d(in_channels=600, out_channels=300, kernel_size=[1, 5], stride=2, bias=True),
        #     nn.ConvTranspose2d(in_channels=300, out_channels=1, kernel_size=[config.embedding_dimension, 5], stride=2, bias=True)
        # )
        self.decoder1 = nn.ConvTranspose2d(in_channels=500, out_channels=600, kernel_size=[1, 12], stride=1, bias=True)
        self.decoder2 = nn.ConvTranspose2d(in_channels=600, out_channels=300, kernel_size=[1, 5], stride=2, bias=True)
        self.decoder3 = nn.ConvTranspose2d(in_channels=300, out_channels=1, kernel_size=[config.embedding_dimension, 5], stride=2, bias=True)

        self.flatten = nn.Flatten()
        self.fc = nn.Linear(500, 2)  # 全连接层

    def forward(self, x, useDCNN):
        # [32, 30]
        x = self.embedding(x)

        # [32, 60, 300]
        x = x.unsqueeze(0)
        # x = x.permute(2, 0, 3, 1)

        # [1, 32, 60, 300]
        x = x.permute(1, 0, 3, 2)
        # [1, 32, 300, 60]
        # x = self.encoders(x)
        # RuntimeError: Given groups=1, weight of size [300, 1, 300, 5],
        # expected input[1, 32, 60, 300] to have 1 channels, but got 32 channels instead

        # x = x.type(torch.cuda.FloatTensor)
        x = self.encoder1(x)
        x = self.encoder2(x)
        x = self.encoder3(x)

        if (useDCNN):
            # x = self.decoders(x)
            x = self.decoder1(x)
            x = self.decoder2(x)
            x = self.decoder3(x)
        else:
            # [32, 500, 1, 1]
            x = x.squeeze(2)
            x = self.flatten(x)
            # [32, 500]
            x = self.fc(x)

        return x


