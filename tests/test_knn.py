import numpy as np
from reviews.models import KNNClassifier


def test_KNN():
    '''
    In this test, we look at the closest vector. 
    Notice that the vectors that KNN looks at are not
    the ones in the matrix, but their normalizations. 
    Since the positive and negative ones are very far
    from each other, and each one has a vector of the same
    class close to it, the score should be 1.
    '''
    # The matrix to set the class.
    X = np.array([[1, 1],
                 [0, 1],
                 [1, 0],
                 [-1, 0],
                 [0, -1]])
    # The classification of the vectors.
    y = np.array([[1],
                  [1],
                  [1],
                  [0],
                  [0]])

    knn = KNNClassifier(1)  # Set the model.
    knn.fit(X, y)  # Fit the model.
    score = knn.score(X, y)  # Calculate the score.
    assert score == 1.0


def test_KNN_2():
    '''
    In this test, we look at the 3 closest vectors. 
    Notice that the vectors that KNN looks at are not
    the ones in the matrix, but their normalizations. 
    Since we are looking at 3 vectors and not just one,
    and the vectors of different categories are far from
    each other and there are just 2 negative ones, the positive
    ones should be predicted correctly but the negative ones
    should not. So the score predicted should be 0.6.
    '''
     # The matrix to set the class.
    X = np.array([[1, 1],
                 [0, 1],
                 [1, 0],
                 [-1, 0],
                 [0, -1]])
    # The classification of the vectors.
    y = np.array([[1],
                  [1],
                  [1],
                  [0],
                  [0]])

    knn = KNNClassifier(3)  # Set the model.
    knn.fit(X, y)  # Fit the model.
    score = knn.score(X, y)  # Calculate the score.
    assert score == 0.6
