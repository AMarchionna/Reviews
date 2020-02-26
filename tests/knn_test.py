import numpy as np
import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(0, '/home/agustin/git_repo/Reviews')
from reviews.models import KNNClassifier
sys.path.insert(0, '~/git_repo/Reviews/tests')


def test_KNN():
    X = np.array([[1, 1],
                 [0, 1],
                 [1, 0],
                 [-1, 0],
                 [0, -1]])

    y = np.array([[1],
                  [1],
                  [1],
                  [0],
                  [0]])

    knn = KNNClassifier(1)
    knn.fit(X, y)
    score = knn.score(X, y)
    assert score == 1.0


def test_KNN_2():
    X = np.array([[1, 1],
                 [0, 1],
                 [1, 0],
                 [-1, 0],
                 [0, -1]])

    y = np.array([[1],
                  [1],
                  [1],
                  [0],
                  [0]])

    knn = KNNClassifier(3)
    knn.fit(X, y)
    score = knn.score(X, y)
    assert score == 0.6
