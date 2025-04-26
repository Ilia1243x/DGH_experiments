import numpy as np

from constants import DEFAULT_SEED


def rnd_R(n, m, rnd=None):
    """
    Generates random vertex of bi-mapping polytope 𝓢.

    :param n: cardinality of X (int)
    :param m: cardinality of Y (int)
    :param rnd: NumPy random state
    :return: mapping pair R ∈ 𝓡  (2d-array)
    """
    rnd = rnd or np.random.RandomState(DEFAULT_SEED)

    R = np.zeros(n+m, n+m)
    R[np.arange(n), rnd.choice(m, n)] = 1
    R[n + np.arange(m), m + rnd.choice(n, m)] = 1

    return R


def rnd_S(n, m, rnd=None):
    """
    Generates random soft mapping in X🠖Y as a point in the bi-mapping polytope 𝓢.

    :param n: cardinality of X (int)
    :param m: cardinality of Y (int)
    :param rnd: NumPy random state
    :return: soft mapping pair S ∈ 𝓢  (2d-array)
    """
    rnd = rnd or np.random.RandomState(DEFAULT_SEED)
    nxn_zeros, mxm_zeros = (np.zeros((size, size)) for size in [n, m])

    # Generate random n×m and m×n row-stochastic matrices.
    F_soft, G_soft = (rnd.rand(size1, size2)
                        for size1, size2 in [(n, m), (m, n)])
    for soft in (F_soft, G_soft):
        soft /= soft.sum(axis=1)[:, None]

    S = np.block([[F_soft, nxn_zeros], [mxm_zeros, G_soft]])

    return S


def center(n, m):
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


def is_row_stoch(S):
    """
    Checks if S is row-stochastic.

    :param S: 2d-array
    :return: bool
    """
    return np.allclose(np.sum(S, axis=1), 1) and ((0 <= S) & (S <= 1)).all()


def is_in_bimapping_polytope(S, n, m):
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
    nxn_zeros, mxm_zeros = (np.zeros((size, size), dtype=int) for size in [n, m])

    # Construct matrix representations of the mappings.
    F = np.identity(m)[f]
    G = np.identity(n)[g]

    R = np.block([[F, nxn_zeros], [mxm_zeros, G]])

    return R


def S_to_fg(S, n, m):
    """
    Projects a soft mapping pair onto the space of mapping pairs.

    :param S: soft mapping pair S ∈ 𝓢  (2d-array)
    :param n: cardinality of X (int)
    :param m: cardinality of Y (int)
    :return: projections f:X🠖Y (1d-array), g:Y🠖X (1d-array)
    """
    f = np.argmax(S[:n, :m], axis=1)
    g = np.argmax(S[n:, m:], axis=1)

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
    #print(S)
    f, g = S_to_fg(S, n, m)
    R = fg_to_R(f, g)

    return R

import numpy as np

def project_row_simplex(row):
    """
    Проецирует вектор на вероятностьный симплекс.

    Аргументы:
    - row: 1D массив (не обязательно нормированный)

    Возвращает:
    - Спроецированный вектор на симплекс
    """
    sorted_row = np.sort(row)[::-1]
    cumulative_sum = np.cumsum(sorted_row)
    
    k = np.argmax(sorted_row - (cumulative_sum - 1) / (np.arange(1, len(row) + 1)) > 0) + 1
    tau = (cumulative_sum[k - 1] - 1) / k
    
    return np.maximum(row - tau, 0)

def proj_simplex(S,n,m):
    S[:n, m:] = 0
    S[n:, :m] = 0
    S[n:, m:] = proj_block(S[n:, m:])
    S[:n, :m] = proj_block(S[:n, :m])
    return S

def proj_block(S):
    sorted_S = np.sort(S, axis=1)[:, ::-1]  
    cumsum_S = np.cumsum(sorted_S, axis=1)

    k = np.arange(1, S.shape[1] + 1)
    tau_candidates = (cumsum_S - 1) / k
    valid = sorted_S - tau_candidates > 0
    rho = valid.sum(axis=1) - 1
    tau = tau_candidates[np.arange(S.shape[0]), rho]

    return np.maximum(S - tau[:, None], 0)
