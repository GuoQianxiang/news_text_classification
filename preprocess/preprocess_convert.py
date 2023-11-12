import chardet
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from sklearn.preprocessing import OneHotEncoder


# Calculate sentence vectors by word vectors in different ways
def calculate_sentence_vec(sentence, model, vector_size, pattern):
    # initialize sentence vector as all zero vector
    vector = np.zeros(vector_size)
    # split words by " "
    words = sentence.split()
    if pattern == 'sum':
        # summary all word vectors as final sentence vector
        for word in words:
            vector = vector + model.wv[word]
    elif pattern == 'average':
        for word in words:
            vector = vector + model.wv[word]
        # average all vectors
        vector = vector / len(words)
    elif not isinstance(pattern, str):
        for i in range(len(words)):
            vector = vector + model.wv[words[i]] * pattern[i]
    else:
        print('You must give the method you want to calculate the sentence vectors!!!')
    return vector


def convert_sentence_to_vector(df, column_name, model, vector_size):
    # Define sentences which will be used in model fitting
    sentences = []
    print('Splitting sentences to words in %s column...' % column_name)
    for text in df[column_name]:
        sentences.append(text.split())

    # fit the model such as Word2Vec
    my_model = model(sentences, vector_size=vector_size, window=5, min_count=1, workers=4)

    print('Converting natural language to sentence vectors in %s column...' % column_name)
    df[column_name] = df[column_name].apply(lambda x: calculate_sentence_vec(str(x), my_model, vector_size, 'sum'))


# 将数据进行onehot编码
def convert_word_to_onehot(df, column_name):
    # Initialize one-hot model
    encoder = OneHotEncoder(sparse_output=False)
    encoder.fit(df[column_name].values.reshape(-1, 1))
    # Convert words to vector
    one_hot_encoded = encoder.transform(df[column_name].values.reshape(-1, 1))
    print('Converting word to onehot vectors in %s column...' % column_name)
    df[column_name] = pd.Series(one_hot_encoded.tolist())


# 通过panda读取csv文件
def convert_data_to_vectors(data, columns_to_onehot, columns_to_vector):

    for column in columns_to_onehot:
        convert_word_to_onehot(data, column)

    for column in columns_to_vector.keys():
        convert_sentence_to_vector(data, column, Word2Vec, columns_to_vector[column])

    return data
