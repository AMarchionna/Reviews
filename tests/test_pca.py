import numpy as np
from reviews.models import PCA


def test_PCA():
    '''
    In this test we check that the PCA correctly
    transforms the given matrix completely, that is,
    obtains the two eigenvectors of the covariant 
    matrix (a 2x2 matrix) and does the multiplication 
    correctly.
    '''
    # Matrix we want to transform.
    X = np.array([[2, 1],
                 [3, 4],
                 [5, 0],
                 [7, 6],
                 [9, 2]])

    pca = PCA(2)  # Set a PCA model with one component.
    pca.fit(X)  # Fit the model to the given matrix.

    X = pca.transform(X)  # Transform the given matrix according to the model.

    # The matrix we should approximately obtain.
    X_ac = np.array([[-3.578, 0],
                     [-1.342, 2.236],
                     [-1.342, -2.236],
                     [3.13, 2.236],
                     [3.13, -2.236]])
    for i in range(len(X)):  # We iterate both matrices.
        for j in range(len(X[i])):
            # We verify that they are the same within some epsilon difference.
            assert abs(X[i][j] - X_ac[i][j]) < 0.001


def test_PCA_2():
    '''
    In this test we check that the PCA correctly
    transforms the given matrix but taking into account
    just one eigenvector of the covariant matrix (the one
    corresponding to the largest eigenvalue of the 2, as 
    it is a 2x2 matrix) and does the multiplication 
    correctly.
    '''
    # Matrix we want to transform.
    X = np.array([[2, 1],
                  [3, 4],
                  [5, 0],
                  [7, 6],
                  [9, 2]])

    pca = PCA(1)  # Set a PCA model with one component.
    pca.fit(X)  # Fit the model to the given matrix.

    X = pca.transform(X)  # Transform the given matrix according to the model.

    # The matrix we should approximately obtain.
    X_ac = np.array([[-3.578],
                     [-1.342],
                     [-1.342],
                     [3.13],
                     [3.13]])
    for i in range(len(X)):  # We iterate both matrices.
        for j in range(len(X[i])):
            # We verify that they are the same within some epsilon difference.
            assert abs(X[i][j] - X_ac[i][j]) < 0.001
