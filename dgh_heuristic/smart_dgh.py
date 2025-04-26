import numpy as np
import random

from mappings import rnd_S, center, S_to_fg, S_to_R, is_in_bimapping_polytope
from fw import make_frank_wolfe_solver
from spaces import diam, rad, arrange_distances
from constants import DEFAULT_SEED, C_SEARCH_GRID
from dgh import dis, upper


def fg_to_R_float(f, g):
    """
    Represents a mapping pair as a binary row-stochastic block matrix.

    :param f: mapping in X🠖Y (1d-array)
    :param g: mapping in Y🠖X (1d-array)
    :return: mapping pair representation R ∈ ℛ (2d-array)
    """
    n, m = len(f), len(g)
    nxn_zeros, mxm_zeros = (np.zeros((size, size), dtype=float) for size in [n, m])

    F = np.identity(m, dtype=float)[f]
    G = np.identity(n, dtype=float)[g]

    R = np.block([[F, nxn_zeros], [mxm_zeros, G]])

    return R

def update_fg(best_f, best_g, set_f, set_g, change_number = 1):

    f = best_f.copy()
    g = best_g.copy()

    pairs_f = random.sample(set_f, min(len(set_f),change_number))
    pairs_g = random.sample(set_g, min(len(set_g),change_number))

    for pair_f in pairs_f:
       if (pair_f[0] != pair_f[1]):
          continue
       new_destination = np.random.choice(len(g), 2, replace=False)
       f[pair_f[0]],f[pair_f[1]] = new_destination[0], new_destination[1]
    
    for pair_g in pairs_g:
       if (pair_g[0] != pair_g[1]):
          continue
       new_destination = np.random.choice(len(f), 2, replace=False)
       g[pair_g[0]],g[pair_g[1]] = new_destination[0], new_destination[1]

    return f,g
    
def calculate_dgh(X, Y, budget_for_start = 1000, search_budget = 2000, change_number = 1, tol = 1e-16, return_fg = False):

    n, m = len(X), len(Y)
    
    diam_X, diam_Y = map(diam, [X, Y])
    d_max = max(diam_X, diam_Y)

    rad_X, rad_Y = map(rad, [X, Y])
    lb = max(0, abs(diam_X - diam_Y)/2, abs(rad_X - rad_Y)/2)

    X, Y = map(lambda Z: Z.copy() / d_max, [X, Y])
    lb /= d_max

    start_dgh, best_f, best_g, c = upper(X,Y,iter_budget=budget_for_start, return_fg = True)

    start_dgh *= d_max
    best_dis_R = start_dgh * 2

    if np.isclose(best_dis_R, lb, atol=0.1): #atol = 1?
        res = (best_dis_R/2, best_f, best_g) if return_fg else best_dis_R/2
        return res
    
    set_f = []
    set_g = []

    for i in range(len(best_f)):
       for j in range(i+1, len(best_f)):
          if best_f[i] == best_f[j]:
             set_f.append((i, j))
    
    for i in range(len(best_g)):
       for j in range(i+1, len(best_g)):
          if best_g[i] == best_g[j]:
             set_g.append((i, j))

    fw = make_frank_wolfe_solver(X, Y, c, tol=tol)

    while (search_budget > 0):

        f, g = update_fg(best_f, best_g, set_f, set_g, change_number)

        S0 = fg_to_R_float(f, g)
        
        S, used_iter = fw(S0=S0, max_iter=search_budget)

        search_budget -= used_iter

        R = S_to_R(S, n, m)
        dis_R = dis(R, X, Y) * d_max
        
        print(dis_R, best_dis_R)
        
        if dis_R != best_dis_R:
            print(dis_R, best_dis_R)

        if dis_R < best_dis_R:
            best_f, best_g = map(list, S_to_fg(S, n, m))
            best_dis_R = dis_R
        
        if np.isclose(best_dis_R, lb, atol=0.1): #atol = 1?
           break
    
    res = (best_dis_R/2, best_f, best_g) if return_fg else best_dis_R/2

    return res