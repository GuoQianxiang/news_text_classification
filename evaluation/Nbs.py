import utils
from model.NBS import NBC
from sklearn.metrics import f1_score
import utils.dataloader as dataloader
from sklearn.naive_bayes import GaussianNB


if __name__ == '__main__':

    training_data = dataloader.load('../preprocess/converted_training.csv')
    validation_data = dataloader.load('../preprocess/converted_validation.csv')
    # split training data and label
    x_train = training_data[:, :-1]
    y_train = training_data[:, -1]
    # shuffle the training data and split to train and test part
    # x_train, y_train, x_test, y_test = utils.shuffle_data(x_train, y_train)

    x_test = validation_data[:, :-1]
    y_test = validation_data[:, -1]

    model = NBC(5)  # instance of NBC
    # model = GaussianNB()
    model.fit(x_train, y_train)  # train model
    y_prediction = model.predict(x_test)

    f1 = f1_score(y_test, y_prediction, average='macro')
    print("F1-score: {:.2f}".format(f1))
    print("Accuracy:", model.score(x_test, y_test))


