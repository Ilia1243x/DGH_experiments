import numpy as np
import torch
from itertools import permutations

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def diam(X):
    """
    Finds the diameter of a metric space X.

    :param X: distance matrix of X (2d-array)
    :return: diameter (float)
    """
    return X.max().item()


def rad(X):
    """
    Finds the radius of a metric space X.

    :param X: distance matrix of X (2d-array)
    :return: radius (float)
    """
    return torch.min(torch.max(X, dim=0).values)


def validate_tri_ineq(X): #не трогай (это на нг)
    """
    Validates that the distance on X obeys the triangle inequality.

    :param X: distance matrix of X (2d-array)
    :return: whether the triangle inequality holds (bool)
    """
    for i, j, k in permutations(range(len(X)), 3):
        if X[i, j] > X[j, k] + X[k, i]:
            return False

    return True


def arrange_distances(X, Y):
    """
    Arranges distances of X and Y in block matrices used in the computations.

    :param X: distance matrix of X (2d-array)
    :param Y: distance matrix of Y (2d-array)
    :return: three block matrices
    """
    nxm_zeros = torch.zeros((X.size(0), Y.size(0)), dtype=torch.int, device = device)

    X__Y = torch.cat([
        torch.cat([X, nxm_zeros], dim = 1),
        torch.cat([nxm_zeros.T, Y], dim = 1)
    ], dim = 0)

    Y__X = torch.cat([
        torch.cat([Y, nxm_zeros.T], dim = 1),
        torch.cat([nxm_zeros, X], dim = 1)
    ], dim = 0)

    _Y_X = torch.cat([
        torch.cat([nxm_zeros.T, Y], dim = 1),
        torch.cat([X, nxm_zeros], dim = 1)
    ], dim = 0)

    return X__Y, Y__X, _Y_X
