import pandas as pd
import chardet
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


# 删除停用词的函数
def remove_stopwords(text):
    stop_words = set(stopwords.words('english'))  # 加载停用词列表
    word_tokens = word_tokenize(text)  # 将文本拆分为单词
    filtered_words = [word for word in word_tokens if word.lower() not in stop_words]  # 过滤停用词
    return filtered_words


# 读取csv文件并进行预处理
def preprocess_csvfile(filename):
    # 不确定编码方式，尝试使用chardet库来检测编码方式
    rawdata = open(filename, 'rb').read()
    result = chardet.detect(rawdata)
    encoding = result['encoding']

    # 通过panda读取csv文件
    data = pd.read_csv(filename, encoding=encoding)
    data = data.drop(columns='id')  # 去掉id列
    data = data.dropna()  # 去掉空行

    # 删除停用词：出现频率高，但是贡献不高
    data['T1'] = data['T1'].apply(remove_stopwords)
    data['T2'] = data['T2'].apply(remove_stopwords)

    # 删除标点符号等
    data['T1'] = data['T1'].apply(lambda x: re.sub('[^a-zA-Z]', ' ', str(x)))
    data['T1'] = data['T1'].apply(lambda x: re.sub('\s[a-zA-Z]\s', '', str(x)))
    data['T1'] = data['T1'].apply(lambda x: re.sub('\s+', ' ', str(x)))
    data['T1'] = data['T1'].apply(lambda x: re.sub('\s[a-zA-Z]\s', '', str(x)))
    data['T2'] = data['T2'].apply(lambda x: re.sub('[^a-zA-Z]', ' ', str(x)))
    data['T2'] = data['T2'].apply(lambda x: re.sub('\s[a-zA-Z]\s', '', str(x)))
    data['T2'] = data['T2'].apply(lambda x: re.sub('\s+', ' ', str(x)))
    data['T2'] = data['T2'].apply(lambda x: re.sub('\s[a-zA-Z]\s', '', str(x)))

    # 转换成小写的书写形式
    data['T1'] = data['T1'].str.lower()
    data['T2'] = data['T2'].str.lower()

    data.to_csv('preprocessed_' + filename, index=True)

    # return data