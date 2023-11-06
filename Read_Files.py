import chardet
import numpy as np
import pandas as pd
import torch
from gensim.models import Word2Vec
from sklearn.preprocessing import OneHotEncoder


def calculate_sentence_vec(sentence, model, vector_size):
    # initialize sentence vector as all zero vector
    vector = np.zeros(vector_size)
    # split words by " "
    words = sentence.split()
    # summary all word vectors as final sentence vector
    for word in words:
        vector = vector + model.wv[word]
    return vector


def convert_nl_to_vector(df, column_name, model, vector_size):
    sentences = []
    for text in df[column_name]:
        sentences.append(text.split())
    print(sentences[0])
    my_model = model(sentences, vector_size=vector_size, window=5, min_count=1, workers=4)
    # my_model.wv('grasp')
    df[column_name] = df[column_name].apply(lambda x: calculate_sentence_vec(str(x), my_model, vector_size))


# 将数据进行onehot编码
def convert_word_by_onehot(df, column_name):
    # Initialize one-hot model
    encoder = OneHotEncoder(sparse_output=False)
    encoder.fit(df[column_name].values.reshape(-1, 1))
    # Convert words to vector
    one_hot_encoded = encoder.transform(df[column_name].values.reshape(-1, 1))
    df[column_name] = pd.Series(one_hot_encoded.tolist())


# 通过panda读取csv文件
def convert_file_to_vector(filename):
    # 不确定编码方式，尝试使用chardet库来检测编码方式
    rawdata = open(filename, 'rb').read()
    result = chardet.detect(rawdata)
    encoding = result['encoding']

    #
    preprocessed_training_data = pd.read_csv(filename, encoding=encoding)
    convert_nl_to_vector(preprocessed_training_data, 'T1', Word2Vec, 15)
    convert_nl_to_vector(preprocessed_training_data, 'T2', Word2Vec, 25)
    convert_nl_to_vector(preprocessed_training_data, 'S', Word2Vec, 10)
    convert_word_by_onehot(preprocessed_training_data, 'TO')

    preprocessed_training_data.to_csv('Converted_' + filename, index=True)


convert_file_to_vector('preprocessed_training_data.csv')
# convert_file_to_vector('pre.csv')
# convert_file_to_vector('test.csv')
