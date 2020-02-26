import numpy as np
import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(0, '/home/agustin/git_repo/Reviews')
from reviews.models import PCA
sys.path.insert(0, '~/git_repo/Reviews/tests')


def test_PCA():
    X = np.array([[2, 1],
                 [3, 4],
                 [5, 0],
                 [7, 6],
                 [9, 2]])

    pca = PCA(2)
    pca.fit(X)

    X = pca.transform(X)

    X_ac = np.array([[-3.578, 0],
                     [-1.342, 2.236],
                     [-1.342, -2.236],
                     [3.13, 2.236],
                     [3.13, -2.236]])
    print(X)
    for i in range(len(X)):
        for j in range(len(X[i])):
            assert abs(X[i][j] - X_ac[i][j]) < 0.001


def test_PCA_2():
    X = np.array([[2, 1],
                  [3, 4],
                  [5, 0],
                  [7, 6],
                  [9, 2]])

    pca = PCA(1)
    pca.fit(X)

    X = pca.transform(X)

    X_ac = np.array([[-3.578],
                     [-1.342],
                     [-1.342],
                     [3.13],
                     [3.13]])
    print(X)
    for i in range(len(X)):
        for j in range(len(X[i])):
            assert abs(X[i][j] - X_ac[i][j]) < 0.001
