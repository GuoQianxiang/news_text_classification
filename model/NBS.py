import math
import numpy as np


class NBC:
    def __init__(self, num_classes):
        self.num_labels = num_classes
        self.model = None

    # Calculate the expectation
    @staticmethod
    def mean(x):
        return sum(x) / float(len(x))

    # Calculate the standard deviation
    def stdev(self, x):
        avg = self.mean(x)
        return math.sqrt(sum([math.pow(i - avg, 2) for i in x]) / len(x))

    # Gaussian probable density function
    def gaussian_probability(self, x, mean, stdev):
        ex = math.exp(-(pow(x - mean, 2) / (2 * pow(stdev, 2))))
        return (1 / math.sqrt(2 * math.pi * pow(stdev, 2))) * ex

    # Handle X_train
    def summarize(self, train_data):
        summaries = [(self.mean(i), self.stdev(i)) for i in zip(*train_data)]
        return summaries

    # Calculate the mathematical expectation and standard deviation respectively
    def fit(self, x, y):
        # Take unrepeatable values of y
        labels = list(set(y))
        # Verify the data
        if len(labels) != self.num_labels:
            print('The number of feature types in the real data is inconsistent with the initial number!!')
        # Defines a collection ready to select data by different labels
        data = {label: [] for label in labels}
        for f, label in zip(x, y):
            data[label].append(f)
        # print(data.items())
        self.model = {label: self.summarize(value) for label, value in data.items()}
        # print(self.model.items())
        return "ok"

    # Calculate the posterior probabilities to all labels
    def calculate_probabilities(self, input_data):
        probabilities = {}
        for label, value in self.model.items():
            probabilities[label] = 0
            for i in range(len(value)):
                mean, stdev = value[i]
                probabilities[label] += np.log(self.gaussian_probability(input_data[i], mean, stdev))
        return probabilities

    # Predict the label of data
    def predict(self, x_test):
        # Sort the probability of the predicted data in all categories and take the highest one
        prediction = []
        for i in x_test:
            label = sorted(self.calculate_probabilities(i).items(), key=lambda x: x[-1])[-1][0]
            prediction.append(label)
        return prediction

    # Calculate the accuracy of model
    def score(self, x_test, y_test):
        prediction = self.predict(x_test)
        acc = np.mean(prediction == y_test)
        return acc
