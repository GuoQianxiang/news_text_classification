import numpy as np
import pandas as pd


def preprocess(data, columns_to_onehot, columns_to_vector):
    for column in columns_to_onehot:
        data[column] = data[column].apply(lambda x: np.array(x.replace('\n', '').lstrip('[').rstrip(']').split(',')))

    for column in columns_to_vector.keys():
        data[column] = data[column].apply(lambda x: np.array(x.replace('\n', '').lstrip('[').rstrip(']').split()))

    return data


def load(filename):
    data = pd.read_csv(filename)
    columns_to_converted = ['T0', 'T1', 'T2', 'S']
    columns_to_onehot = ['T0']
    columns_to_vector = {'T1': 15, 'T2': 25, 'S': 10}
    data = preprocess(data, columns_to_onehot, columns_to_vector)
    vector_size = 21  # include two discrete value and one label
    vector_num = data.shape[0]
    for column in columns_to_converted:
        vector_size = vector_size + data[column][1].shape[0]
    print('vector_size is', vector_size)
    print('vector_num is', vector_num)

    data = data.to_numpy()

    final_data = np.zeros((vector_num, vector_size))
    for i in range(vector_num):
        final_data[i] = np.hstack(
            (data[i][0], data[i][1], data[i][2], data[i][3],
             np.repeat(data[i][4], 10),
             np.repeat(data[i][5], 10),
             data[i][6]))

    return final_data


def predict(test_file, model):
    test_data = load(test_file)
    test_label = model.predict(test_data[:, :-1])

    predict_result = pd.read_csv(test_file)
    predict_result['class label'] = test_label
    return predict_result


def save_test(test_data, model_name):
    test_data.to_csv('test_predicted_result_' + model_name + '.csv', index=False)
    print('test_data has been saved as test_predicted_result_%s.csv!!!' % model_name)
