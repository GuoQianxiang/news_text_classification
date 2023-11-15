import pandas as pd
import chardet
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


def open_file(filename):
    print('Opening file...,file name is:', filename)
    # 不确定编码方式，尝试使用chardet库来检测编码方式
    rawdata = open(filename, 'rb').read()
    result = chardet.detect(rawdata)
    encoding = result['encoding']
    data = pd.read_csv(filename, encoding=encoding)
    return data


# 删除停用词的函数
def search_stopwords(text):
    stop_words = set(stopwords.words('english'))  # 加载停用词列表
    word_tokens = word_tokenize(str(text))  # 将文本拆分为单词
    filtered_words = [word for word in word_tokens if word.lower() not in stop_words]  # 过滤停用词
    return filtered_words


# 读取csv文件并进行预处理
def remove_stopwords(data, column_names, is_test):
    # delete id column and black row
    print('Dropping black row and id column...')
    data = data.drop(columns='id')  # 去掉id列
    if not is_test:
        data = data.dropna()  # 去掉空行

    # remove stopwords which are useless in text
    for column_name in column_names:
        print('Removing stopwords from %s column...' % column_name)
        data[column_name] = data[column_name].apply(search_stopwords)
        # print('Removed stopwords from %s column!!!' % column_name)

    return data


def remove_punctuations(data, column_names):

    patterns = ['[^a-zA-Z]', '\s[a-zA-Z]\s', '\s+', '\s[a-zA-Z]\s']

    # delete all single letter from a~z(including upper case)
    for column_name in column_names:
        print('Deleting useless single letters in %s column...' % column_name)

        for pattern in patterns:
            data[column_name] = data[column_name].apply(lambda x: re.sub(pattern, ' ', str(x)))
            # convert all letters to lower case
            data[column_name] = data[column_name].str.lower()

        # print('Deleted all useless single letters in %s column!!!' % column_name)

    return data


def save_data(data, wanted_filename):
    data.to_csv(wanted_filename, index=False)
    print('Saved cleaned result as %s in preprocess directory!!! Please check it!!!' % wanted_filename)

