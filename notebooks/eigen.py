import numpy as np
import random


def power_iteration(X, num_iter, eps):
    v = np.array([random.random() for i in range(len(X[0]))])
    eigenvalue = 0
    for iteration in range(num_iter):
        mult = X @ v.transpose()
        ev_new = np.linalg.norm(mult) / np.linalg.norm(v)
        if abs(ev_new - eigenvalue) < eps:
            break
        v = X @ v
        v = v / np.linalg.norm(v)

    eigenvalue = (v @ X @ v.transpose()) / (v @ v.transpose())
    v = v / np.linalg.norm(v)

    return (eigenvalue, v)


def get_first_eigenvalues(X, num, num_iter, epsilon):
    A = X
    eigvalues = np.array([-1.0 for i in range(num)])
    eigvectors = np.array([eigvalues for i in range(len(X))])

    for k in range(num):
        eigvalue, eigvector = power_iteration(A, num_iter, epsilon)

        eigvalues[k] = eigvalue
        for i in range(len(eigvectors[0])):
            eigvectors[k][i] = eigvector[i]
        eigvector = eigvector / np.linalg.norm(eigvector)
        eigvector = np.array([eigvector])
        A = A - eigvalue * (eigvector.transpose() @ eigvector)

    return (eigvalues, eigvectors)
