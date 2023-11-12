import utils
import model.dataloader as dataloader
from sklearn.metrics import f1_score
from model.Knn import KNN

if __name__ == '__main__':
    training_data = dataloader.load('../preprocess/converted_training_data.csv')
    # 对引入的数据按照数据和标签进行切割
    x = training_data[:, :-1]  # 得到训练集的数据
    y = training_data[:, -1]  # 得到训练集的标签
    # shuffle the training data and split to train and test part
    x_train, y_train, x_test, y_test = utils.shuffle_data(x, y)

    model = KNN(3)
    model.fit(x_train, y_train)  # 训练模型
    y_prediction = model.predict(x_test)

    f1 = f1_score(y_test, y_prediction, average='macro')
    print("F1-score: {:.2f}".format(f1))