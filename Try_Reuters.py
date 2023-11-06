from keras.datasets import reuters
import nltk
from nltk.corpus import LazyCorpusLoader, CategorizedPlaintextCorpusReader


# https://www.kaggle.com/alvations/testing-1000-files-datasets-from-nltk
reuters = LazyCorpusLoader('reuters', CategorizedPlaintextCorpusReader,
                           '(training|test).*', cat_file='cats.txt', encoding='ISO-8859-2',
                          nltk_data_subdir='/kaggle/input/reuters/reuters/reuters/')
# https://miguelmalvarez.com/2015/03/20/classifying-reuters-21578-collection-with-python-representing-the-data/
reuters.words()

# #查看路透社数据集内容-是已经整型化后的内容
# (train_data, train_lables) , (test_data, test_labels) = reuters.load_data(num_words = 10000)
# # print(train_data[10])
# # print(train_lables[10])

# #查看颠倒后的字典，和解码后的评论
# word_index = reuters.get_word_index()
# reverse_word_index = dict([value, key] for [key, value] in word_index.items())
# decoded_review = ' '.join([reverse_word_index.get(i - 3, '?') for i in train_data[0]])
# print(reverse_word_index)
# print(decoded_review)


