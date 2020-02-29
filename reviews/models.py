import numpy as np


class KNNClassifier:
    '''Class that calculates the nearest vectors to some
        vector and predicts its classification according to
        the classification of the close vectors.'''
    def __init__(self, neighbours):
        '''
        Initates the class with the amount of neighbours to
        look at in the prediction.

        :param neighbours: number of neighbours for the KNN model.
        '''
        self.neighbours = neighbours

    def fit(self, data_train, result_train):
        '''
        Fits the training dataset to the class.

        :param data_train: Dataset to train the model.

        :param result_train: Classification for each vector of dataset.
        '''
        for row in data_train:  # Normalize the vectors.
            row = row / np.linalg.norm(row)
        self.data_train = data_train
        self.result_train = result_train

    def distance_to_row(self, v):
        '''
        Calculates the distance from v to each row of
        of the training dataset and returns a vector of
        the closest indices.

        :param v: The vector to which we calculate the distances.

        :returns: A vector of distances from each row to v.
        '''
        #  Calculate the euclidean distance to each row.
        # Array of pairs(norm, index).
        distances = [(np.linalg.norm((self.data_train)[i] - v, 1), i)
                     for i in range(len(self.data_train))]
        distances.sort()  # Sort with respect to the euclidean distance.
        distances = [b for a, b in distances]  # Keep the indices.
        distances = distances[:self.neighbours]  # Keep the closest indices.
        return distances

    def predict_row(self, v):
        '''
        Predicts a vector and returns its classification predicted.

        :param v: A vector to classify.

        :returns: Either 0 or 1 depending on the classification.
        '''
        distances = self.distance_to_row(v)  # Calculate closest indices.
        # Classify according to indices.
        classify = [self.result_train[i] for i in distances]
        sum = 0  # Calculate the amount of ones predicted.
        for value in classify:
            sum = sum + value
        if 2*sum >= len(classify):  # More ones than zeros predicted.
            return 1
        else:  # More zeros than ones predicted
            return 0

    def predict(self, data_test):
        '''
        Predicts a whole data_set, and returns a vector of the
        classifications predicted.

        :param data_test: The model predicts this dataset.

        :returns: A vector of predictions, one for each vector in the dataset.
        '''
        # Classify each row of the dataset.
        ret = np.array(
            [self.predict_row(data_test[k]) for k in range(len(data_test))])
        return ret

    def score(self, data_test, result_test):
        '''
        Calculates the accuracy of the prediction.

        :param data_test: A dataset to test the model.

        :param result_test: Correct classification of the dataset.

        :returns: Percentage of correct predictions.
        '''
        for row in data_test:  # Normalize dataset
            row = row / np.linalg.norm(row)
        predict_test = self.predict(data_test)  # Predict classification
        assert(len(result_test) == len(predict_test))
        suma = 0.0
        for i in range(len(predict_test)):
            if predict_test[i] == result_test[i]:  # Correct prediction
                suma += 1
        return suma / len(result_test)


class PCA:
    '''Class that calculates the principal components
        of a matrix.'''
    def __init__(self, components):
        '''
        Initiate the class with the number
        of components for PCA.

        :param components: The number of principal components for PCA.
        '''
        self.components = components

    def fit(self, data_train):
        '''
        Calculate the eigenvectors of the covariant
        matrix, to use then in the transform method.

        :param data_train: A dataset to train the model.
        '''
        # Calculate the covariant matrix of the dataset.
        mean = data_train.mean(axis=0)
        A = data_train - mean
        M = np.cov(A.T)
        # Calculate the eigenvalues and eigenvectors
        # of the covariant matrix.
        eigenval, eigenvec = np.linalg.eig(M)
        eigenvec = eigenvec.T
        # Order the eigenvectors from largest eigenvalue to smallest.
        eig = [(eigenval[i], eigenvec[i]) for i in range(len(eigenval))]
        eig.sort(reverse=True)
        for i in range(len(eigenvec)):
            eigenvec[i] = eig[i][1]
        # Keep the eigenvectors corresponding to the largest
        # eigenvalues.
        self.V = eigenvec[:self.components]

    def transform(self, data):
        '''
        Returns the transformed version of the
        dataset, using the calculated eigenvectors.

        :param data: A dataset to transform.

        :returns: The transformed data.
        '''
        mean = data.mean(axis=0)
        data = data - mean
        return data @ self.V.transpose()
