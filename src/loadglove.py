import numpy
import torch
from nltk import word_tokenize
from nltk.corpus import stopwords

def loadGlove(file_loc, file_encoding = 'utf-8'):
    # 加载glove
    # 输入参数：
    # file_loc, str, glove文件的位置
    # file_encodeing, str, glove文件的编码
    # 返回参数：
    # glove_vocab_dic, dic, 词向量词典, word -> index
    # emb, array, index -> vector

    with open(file_loc, 'r', encoding=file_encoding) as file1:
        emb = []
        glove_vocab_dic = {}
        for index, line in enumerate(file1.readlines()):
            row = line.strip().split(' ')
            glove_vocab_dic[row[0]] = index
            emb.append([numpy.float32(x) for x in row[1:]])

    return glove_vocab_dic, emb

def tokenize(sentence, usestopwords = True):
    # 用于将单条句子切割成数组
    # 输入参数：
    # sentence, str, 待转换句子
    # usestopwords, boolean, 是否使用停用词
    # 返回参数
    # words, array, 词数组

    # ****注意****
    # 使用这个前先要使用以下代码
    # nltk.download('stopwords') # 这段用于下载nltk的数据
    words = word_tokenize(sentence)  # 分词
    if usestopwords == True:
        stops = set(stopwords.words("english"))
        words = [word for word in words if word not in stops]
    return words




def sentence2index(sentence, dictionary, unknown = 'unk', minlength = 0, maxlength = 0x3d3d3d):
    # 用于将单条句子转换成index array
    # 输入参数：
    # sentence, str, 待转换句子
    # dictionary, dic, vocab词典
    # unknown, str, 词典没有词汇插入的unk
    # minlength, int, 最终indexes长度最小值限制
    # maxlength, int, 最终indexes长度最大值限制
    # 返回参数
    # indexes, array, index数组

    # 设置unknown
    if unknown not in dictionary and unknown == 'unk':
        unknown = 'UNK'


    indexes = []

    words = tokenize(sentence, usestopwords=True) # 获取分词后的词数组
    counter = 0
    for word in words:
        if word in dictionary: # 看词在不在字典里面
            indexes.append(torch.tensor(dictionary[word], dtype=torch.long))
        else:
            indexes.append(torch.tensor(dictionary[unknown], dtype=torch.long))
        counter += 1
        if counter == maxlength:
            break

    while counter < minlength:
        indexes.append(torch.tensor(dictionary[unknown], dtype=torch.long))
        counter += 1

    return indexes
