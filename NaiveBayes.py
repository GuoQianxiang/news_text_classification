import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB, ComplementNB, CategoricalNB
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score


def preprocess(df):
    df['T1'] = df['T1'].apply(lambda x: np.array(x.replace('\n', '').lstrip('[').rstrip(']').split()))
    df['T2'] = df['T2'].apply(lambda x: np.array(x.replace('\n', '').lstrip('[').rstrip(']').split()))
    df['S'] = df['S'].apply(lambda x: np.array(x.replace('\n', '').lstrip('[').rstrip(']').split(',')))
    df['TO'] = df['TO'].apply(lambda x: np.array(x.replace('\n', '').lstrip('[').rstrip(']').split(',')))
    return df


def concatenate(df):
    result = []
    result = np.concatenate(df['T1'].to_numpy())


def shuffle_data(x, y):
    n, d = x.shape  # Get the rows and columns of data
    n_train = int(0.8 * n)  # Cut 80% of data
    shuffler = np.random.permutation(n)

    x_train = x[shuffler[:n_train]]
    y_train = y[shuffler[:n_train]]
    x_test = x[shuffler[n_train:]]
    y_test = y[shuffler[n_train:]]
    return x_train, y_train, x_test, y_test


# 读取文件方法3：通过pandas导入csv文件
training = pd.read_csv("converted_training_data.csv")
# training_data = preprocess(training)
# validation = pd.read_csv("converted_validation_data.csv")
# print(training['T1'][0].shape)
# print(type(training['T1'].to_numpy()))
# print(training['T1'][0].to_numpy().shape)
# print(training['T1'][0])

# print(type(training_data['T1'][0]))
# print(training_data['T1'][0])
# print(type(training['T1'][0]))
#
# # print(training_data)
# print(type(training_data['T1'][0]))
# print(training_data['T1'])
# print(training_data['T1'])
# print(training_data['T1'].values.reshape(-1, 29993))

# t = np.array(training['T1'][0].replace('\n', '').lstrip('[').rstrip(']').split())
# b = np.array(training['T2'][0].replace('\n', ''))
# print(type(t))
# print(t[0])
# print(t.shape)
# print(b)
# print(np.concatenate(t, b))
# print(type(t[0]))
training_data = preprocess(training)
training_array = training_data.to_numpy()

# 将所有数组连接成一个新的数组
print(len(training_array))
print(training_array[0][2].shape[0])

nums_vector = 3

for i in range(2, 6):
    print(len(training_array[0][i]))
    nums_vector = nums_vector + training_array[0][i].shape[0]

print(nums_vector)
final_training_data = np.zeros((29993, nums_vector))
for i in range(len(training_array)):
    final_training_data[i] = np.hstack((training_array[i][2], training_array[i][3], training_array[i][4],
                                        training_array[i][5], training_array[i][6], training_array[i][7],
                                        training_array[i][8]))

print(final_training_data)
# combined = np.hstack((training_array[0][0], training_array[0][2],training_array[0][3] ,training_array[0][4],training_array[0][5]))
# print(combined)
# print(combined.shape)
# print(training_array.shape)
# print(training_array[0][2].shape)
# print(training_array[0].flatten().shape)
# print(type(training_array[0]))
# print(training_array[0])
# validation_array = validation.to_numpy()

# 对引入的数据按照数据和标签进行切割
x = final_training_data[:, :-1]     #得到训练集的数据
# print(x_train)
# print(training_array)
# # x_test = validation_array[:, :-1]    #得到验证集的数据
y = final_training_data[:, -1]      #得到训练集的标签
# y_test = validation_array[:, -1]     #得到验证集的标签
x_train, y_train, x_test, y_test = shuffle_data(x, y)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

# 建立模型训练并计算正确率
model = GaussianNB()    # 设定高斯分布的朴素贝叶斯分类器
model.fit(x_train, y_train)  # 训练模型
y_pred = model.predict(x_test)


f1 = f1_score(y_test, y_pred, average='macro')
print("F1-score: {:.2f}".format(f1))
