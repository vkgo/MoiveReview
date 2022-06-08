# 加载glove
with open('/dev_data/ts/text_classification/utils/glove.840B.300d.txt', 'r') as file1:
    emb = []
    glove_vocab_dic = {}
    for index, line in enumerate(file1.readlines()):
        row = line.strip().split(' ')
        glove_vocab_dic[row[0]] = index
        emb.append([float(x) for x in row[1:]])

def w2v(data_word, max_len):
    batch_data = []
    for sample_i in range(len(data_word)):
        text_arr = np.zeros((max_len, 300))
        mlen = len(data_word[sample_i]) if len(data_word[sample_i])<max_len else max_len
        for word_j in range(mlen):
            if data_word[sample_i][word_j] in glove_vocab_dic:
                text_arr[word_j] = np.reshape(emb[glove_vocab_dic[data_word[sample_i][word_j]]], [1, 300])
            else:
                text_arr[word_j] = np.reshape(emb[glove_vocab_dic['UNK']], [1, 300])
        batch_data.append(text_arr)
    return np.array(batch_data)