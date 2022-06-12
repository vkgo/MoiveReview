class Config:
    def __init__(self,
                vocabulary_length,  # 词表长度,int
                embedding_dimension,  # 词向量维度,int
                wordvectors  # 词向量，Vectors
                ):
        self.vocabulary_length = vocabulary_length
        self.embedding_dimension = embedding_dimension
        self.wordvectors = wordvectors
