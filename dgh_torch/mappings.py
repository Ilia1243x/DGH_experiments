import numpy as np
import torch

from constants import DEFAULT_SEED

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def rnd_S(n, m, rnd=None):#question
    """
    Generates random soft mapping in X🠖Y as a point in the bi-mapping polytope 𝓢.

    :param n: cardinality of X (int)
    :param m: cardinality of Y (int)
    :param rnd: NumPy random state
    :return: soft mapping pair S ∈ 𝓢  (2d-array)
    """
    rnd = rnd or torch.Generator(device = device).manual_seed(DEFAULT_SEED)
    nxn_zeros, mxm_zeros = (torch.zeros((size, size), device = device) for size in [n, m])

    # Generate random n×m and m×n row-stochastic matrices.
    F_soft, G_soft = (torch.rand((size1, size2), generator = rnd, device = device)
                        for size1, size2 in [(n, m), (m, n)])

    F_soft /= F_soft.sum(dim = 1, keepdim=True)
    G_soft /= G_soft.sum(dim = 1, keepdim=True)

    S = torch.cat([
        torch.cat([F_soft, nxn_zeros], dim = 1),
        torch.cat([mxm_zeros, G_soft], dim = 1)
    ], dim = 0)

    return S


def center(n, m): #пока не трогать
    """
    Returns soft mapping pair that is the barycenter of the bi-mapping polytope 𝓢.

     :param n: cardinality of X (int)
    :param m: cardinality of Y (int)
    :return: barycenter of 𝓢  (2d-array)
    """
    nxn_zeros, mxm_zeros = (np.zeros((size, size), dtype=int) for size in [n, m])

    F_center, G_center = (np.full((size1, size2), 1 / size2)
                            for size1, size2 in [(n, m), (m, n)])

    S = np.block([[F_center, nxn_zeros], [mxm_zeros, G_center]])

    return S


def is_row_stoch(S): #и это тоже 
    """
    Checks if S is row-stochastic.

    :param S: 2d-array
    :return: bool
    """
    return np.allclose(np.sum(S, axis=1), 1) and ((0 <= S) & (S <= 1)).all()


def is_in_bimapping_polytope(S, n, m): #без комментариев 
    """
    Checks if S ∈ 𝓢.

    :param S: 2d-array
    :return: bool
    """
    return is_row_stoch(S[:n, :m]) and is_row_stoch(S[n:, m:]) and \
        np.allclose(S[:n, m:], 0) and np.allclose(S[n:, :m], 0)


def fg_to_R(f, g):
    """
    Represents a mapping pair as a binary row-stochastic block matrix.

    :param f: mapping in X🠖Y (1d-array)
    :param g: mapping in Y🠖X (1d-array)
    :return: mapping pair representation R ∈ ℛ (2d-array)
    """
    n, m = len(f), len(g)
    nxn_zeros, mxm_zeros = (torch.zeros((size, size), dtype=torch.int, device = device) for size in [n, m])

    # Construct matrix representations of the mappings.
    F = torch.eye(m, device = device)[f]
    G = torch.eye(n, device = device)[g]

    R = torch.cat([
        torch.cat([F, nxn_zeros], dim = 1),
        torch.cat([mxm_zeros, G], dim = 1)
    ], dim = 0)

    return R


def S_to_fg(S, n, m):
    """
    Projects a soft mapping pair onto the space of mapping pairs.

    :param S: soft mapping pair S ∈ 𝓢  (2d-array)
    :param n: cardinality of X (int)
    :param m: cardinality of Y (int)
    :return: projections f:X🠖Y (1d-array), g:Y🠖X (1d-array)
    """
    f = torch.argmax(S[:n, :m], dim =1)
    g = torch.argmax(S[n:, m:], dim =1)

    return f, g


def S_to_R(S, n, m):
    """
    Projects a soft mapping pair onto the space of mapping pairs and
    represents the projection as a binary row-stochastic block matrix.

    :param S: soft mapping pair S ∈ 𝓢  (2d-array)
    :param n: cardinality of X (int)
    :param m: cardinality of Y (int)
    :return: mapping pair representation R ∈ ℛ (2d-array)
    """
    f, g = S_to_fg(S, n, m)
    R = fg_to_R(f, g)

    return R
