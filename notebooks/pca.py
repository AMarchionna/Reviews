import numpy as np


class PCA:
    '''Class that calculates the principal components
        of a matrix.'''
    def __init__(self, components):
        '''Initiate the class with the number
            of components for PCA.'''
        self.components = components

    def fit(self, data_train):
        '''Calculate the eigenvectors of the covariant
            matrix, to use then in the transform method.'''
        # Calculate the covariant matrix of the dataset.
        mean = data_train.mean(axis=0)
        A = data_train - mean
        M = np.cov(A.transpose())
        # Calculate the eigenvalues and eigenvectors
        # of the covariant matrix.
        eigenval, eigenvec = np.linalg.eig(M)
        eigenvec = eigenvec.T
        # Order the eigenvectors from largest eigenvalue to smallest.
        eig = [(eigenval[i], eigenvec[i]) for i in range(len(eigenval))]
        eig.sort(reverse=True)
        for i in range(len(eigenvec)):
            a, b = eig[i]
            eigenvec[i] = b
        # Keep the eigenvectors corresponding to the largest
        # eigenvalues.
        self.V = eigenvec[:self.components]

    def transform(self, data):
        '''Returns the transformed version of the
            dataset, using the calculated eigenvectors.'''
        return data @ self.V.transpose()
