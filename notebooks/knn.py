import numpy as np
import random


class KNNClassifier:
    '''Class that calculates the nearest vectors to some
        vector and predicts its classification according to
        the classification of the close vectors.'''
    def __init__(self, neighbours):
        '''Initates the class with the amount of neighbours to
            look at in the prediction.'''
        self.neighbours = neighbours

    def fit(self, data_train, result_train):
        '''Fits the training dataset to the class.'''
        for row in data_train:  # Normalize the vectors.
            row = row / np.linalg.norm(row)
        self.data_train = data_train
        self.result_train = result_train

    def distance_to_row(self, v):
        '''Calculates the distance from v to each row of
            of the training dataset and returns a vector of
            the closest indices.'''
        #  Calculate the euclidean distance to each row.
        # Array of pairs(norm, index).
        distances = [(np.linalg.norm((self.data_train)[i] - v, 1), i)
                     for i in range(len(self.data_train))]
        distances.sort()  # Sort with respect to the euclidean distance.
        distances = [b for a, b in distances]  # Keep the indices.
        distances = distances[:self.neighbours]  # Keep the closest indices.
        return distances

    def predict_row(self, v):
        '''Predicts a vector and returns its classification predicted.'''
        distances = self.distance_to_row(v)  # Calculate closest indices.
        # Classify according to indices.
        classify = [self.result_train[i] for i in distances]
        sum = 0  # Calculate the amount of ones predicted.
        for value in classify:
            sum = sum + value
        if 2*sum > len(classify):  # More ones than zeros predicted.
            return 1
        elif 2*sum < len(classify):  # More zeros than ones predicted
            return 0
        else:  # If there is a tie, return random.
            return random.randint(0, 1)

    def predict(self, data_test):
        '''Predicts a whole data_set, and returns a vector of the
            classifications predicted.'''
        # Classify each row of the dataset.
        ret = np.array(
            [self.predict_row(data_test[k]) for k in range(len(data_test))])
        return ret

    def score(self, data_test, result_test):
        '''Calculates the accuracy of the prediction.'''
        for row in data_test:  # Normalize dataset
            row = row / np.linalg.norm(row)
        predict_test = self.predict(data_test)  # Predict classification
        assert(len(result_test) == len(predict_test))
        suma = 0.0
        for i in range(len(predict_test)):
            if predict_test[i] == result_test[i]:  # Correct prediction
                suma += 1
        return suma / len(result_test)
