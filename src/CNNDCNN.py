from torch import nn


class CNNDCNN(nn.Module):
    # 这是一个3层CNN-DCNN AutoEncoder模型
    def __init__(self, config):
        # config需包含
        # 1. vocabulary_length 2. embedding_dimension 3. wordvectors
        super(CNNDCNN, self).__init__()

        self.embedding = nn.Embedding(config.vocabulary_length, config.embedding_dimension)  # 定义embedding层
        self.embedding = self.embedding.from_pretrained(config.wordvectors)  # 加载词向量


        # CNN Layers
        self.encoders = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=300, kernel_size=[config.embedding_dimension, 5], stride=2, bias=True),
            nn.Conv2d(in_channels=300, out_channels=600, kernel_size=[1, 5], stride=2, bias=True),
            nn.Conv2d(in_channels=600, out_channels=500, kernel_size=[1, 12], stride=1, bias=True)
        )

        # DCNN Layers
        self.decoders = nn.Sequential(
            nn.ConvTranspose2d(in_channels=500, out_channels=600, kernel_size=[1, 12], stride=1, bias=True),
            nn.ConvTranspose2d(in_channels=600, out_channels=300, kernel_size=[1, 5], stride=2, bias=True),
            nn.ConvTranspose2d(in_channels=300, out_channels=1, kernel_size=[config.embedding_dimension, 5], stride=2, bias=True)
        )

    def forward(self, x, useDCNN):
        x = self.embedding(x)  # [1853, 128, 300]

        x = x.unsqueeze(0)  # [1, 1853, 128, 300]

        x = x.permute(2, 0, 3, 1)  # [128, 1, 300, 1853]

        x = self.encoders(x)

        if (useDCNN):
            x = self.decoders(x)

        return x


