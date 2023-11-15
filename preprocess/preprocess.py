import chardet
import pandas as pd
import preprocess_clean as cleaner
import preprocess_convert as converter


def open_file(filename):
    print('Opening file...,file name is:', filename)
    # Uncertain encoding method, try using the chart library to detect encoding method
    rawdata = open(filename, 'rb').read()
    result = chardet.detect(rawdata)
    encoding = result['encoding']
    data = pd.read_csv(filename, encoding=encoding)
    return data


def save_data(data, wanted_filename):
    data.to_csv(wanted_filename, index=False)
    print('Saved data as %s in preprocess directory!!! Please check it!!!' % wanted_filename)


def preprocess_clean(filename, cleaned_filename, is_test):
    data = open_file(filename)
    # Define which columns need to be cleaned
    column_names = ['T1', 'T2']
    data = cleaner.remove_stopwords(data, column_names, is_test)
    data = cleaner.remove_punctuations(data, column_names)
    # Save cleaned data named 'cleaned_filename.csv'
    save_data(data, cleaned_filename)
    return data


def preprocess_convert(filename, converted_filename):
    data = open_file(filename)
    # define which columns need to be converted
    columns_to_onehot = ['T0']
    columns_to_vector = {'T1': 15, 'T2': 25, 'S': 10}
    data = converter.convert_data_to_vectors(data, columns_to_onehot, columns_to_vector)
    save_data(data, converted_filename)
    return data


def preprocess(filename, is_test):
    print('------------------------------------------------------------------------')
    preprocess_clean('../data/' + filename, 'cleaned_' + filename, is_test)
    preprocess_convert('cleaned_' + filename, 'converted_' + filename)
    print('------------------------------------------------------------------------')


if __name__ == '__main__':
    training_file = 'training.csv'
    validation_file = 'validation.csv'
    test_file = 'test.csv'
    preprocess(training_file, False)
    preprocess(validation_file, False)
    preprocess(test_file, True)


