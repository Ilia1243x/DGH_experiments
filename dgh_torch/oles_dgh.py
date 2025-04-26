import numpy as np
import time
import torch

from mappings import rnd_S, center, S_to_fg, S_to_R, is_in_bimapping_polytope
from fw import solve_frank_wolfe
from spaces import diam, rad, arrange_distances
from constants import DEFAULT_SEED, C_SEARCH_GRID

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def dis_oles(S, X__Y, Y__X, _Y_X):
    """
    Calculates "distortion" of a soft mapping pair, which coincides with actual
    distortion on the space of mapping pairs/correspondences.

    :param S: soft mapping pair S ∈ 𝓢  (2d-array)
    :param X: distance matrix of X (2d-array)
    :param Y: distance matrix of Y (2d-array)
    :return: distortion (float)
    """
    
    #S = S.to(device)
    #_Y_X = _Y_X.to(device)
    #X__Y = X__Y.to(device)
    #Y__X = Y__X.to(device)
    
    S_Y_X = torch.mm(S, _Y_X)

    dis_S = torch.abs(X__Y - torch.mm(torch.mm(S, Y__X), S.T) + S_Y_X - S_Y_X.T).max()

    return dis_S


def upper_oles(X, Y, c='auto', iter_budget=100, S0=None, tol=1e-16, return_fg=False,
          lb=0, validate_tri_ineq=False, verbose=0, rnd=None, c_find = False):
    """
    Finds am upper bound of dGH(X, Y) by minimizing smoothed dis(R) = dis(f, g)
    over the bi-mapping polytope 𝓢 using Frank-Wolfe algorithm.

    :param X: distance matrix of X (2d-array)
    :param Y: distance matrix of Y (2d-array)
    :param c: exponentiation base ∈ (1, ∞) for smoothing the distortion
        in the first minimization problem (float)
    :param iter_budget: total number of Frank-Wolfe iterations (int)
    :param S0: first starting point (subsequent restarts always use random ones):
        2d-array, 'center' for the center of 𝓢, None for random point in 𝓢
    :param tol: tolerance to use when evaluating convergence (float)
    :param return_fg: whether to return the optimal pair of mappings (bool)
    :param lb: lower bound of dGH(X, Y) to avoid redundant iterations (float)
    :param validate_tri_ineq: whether to validate the triangle inequality (bool)
    :param verbose: no output if 0, summary if >0, restarts if >1, iterations if >2
    :param rnd: random number generator to use for restarts
    :return: dGH(X, Y), f [optional], g [optional]
    """
    # Ensure positive iteration budget.
    assert iter_budget > 0, 'insufficient iteration budget'

    # Check that the distances satisfy the metric properties minus the triangle inequality.
    assert (X >= 0).all() and (Y >= 0).all(), 'distance matrices have negative entries'
    assert (torch.diag(X) == 0).all() and (torch.diag(Y) == 0).all(),\
        'distance matrices have non-zeros on the main diagonal'
    assert (X == X.T).all() and (Y == Y.T).all(), 'distance matrices are not symmetric'
    if validate_tri_ineq:
        assert validate_tri_ineq(X) and validate_tri_ineq(Y),\
            "triangle inequality doesn't hold"

    # Initialize.
    n, m = X.size(0), Y.size(0)
    rnd = rnd or torch.Generator(device = device).manual_seed(DEFAULT_SEED)
    best_dis_R = float('inf')

    # Update lower bound using the radius and diameter differences.
    diam_X, diam_Y = map(diam, [X, Y])
    rad_X, rad_Y = map(rad, [X, Y])
    lb = max(lb, abs(diam_X - diam_Y)/2, abs(rad_X - rad_Y)/2)

    if verbose > 0:
        print(f'iteration budget {iter_budget} | c={c} | dGH≥{lb}')

    # Search for best c if not specified.
    if c == 'auto':
        start = time.time()
        # Allocate 50% of iteration budget for the search.
        search_iter_budget_per_c = iter_budget // (2*len(C_SEARCH_GRID))
        search_iter_budget = search_iter_budget_per_c * len(C_SEARCH_GRID)
        iter_budget -= search_iter_budget

        # Select c resulting in the smallest upper bound.
        init_rnd_state = rnd.get_state()
        for c_test in C_SEARCH_GRID:
            rnd.set_state(init_rnd_state)
            ub = upper_oles(X, Y, c=c_test, iter_budget=search_iter_budget_per_c,
                             S0=S0, tol=tol, return_fg=True, lb=lb, rnd=rnd, c_find=True)
            if ub < best_dis_R/2:
                c = c_test
                rnd_state = rnd.get_state()
                best_dis_R = 2*ub

        # Set random number generator to after the search iterations.
        rnd.set_state(rnd_state)
        print(f'time on c: {time.time()-start}')

    # Scale all distances to avoid overflow.
    d_max = max(diam_X, diam_Y)
    X, Y = map(lambda Z: Z.clone() / d_max, [X, Y])
    lb /= d_max

    # Find minima from new restarts until iteration budget is depleted.
    restart_idx = 0
   
    X__Y, Y__X, _Y_X = arrange_distances(X, Y)

    #X__Y, Y__X, _Y_X = torch.from_numpy(X__Y), torch.from_numpy(Y__X), torch.from_numpy(_Y_X)
    
    c_Y_X, c__Y_X = c**_Y_X,  c**-_Y_X
    c__X__Y, c_Y__X = c**-X__Y, c**Y__X
    c_X__Y, c__Y__X =  c**X__Y, c**-Y__X

    start = time.time()

    time_dot = 0.0
    time_R = 0.0
    time_check = 0.0
    
    while iter_budget > 0:
        
        # Initialize new restart.
        if restart_idx > 0 or S0 is None:
            S0 = rnd_S(n, m, rnd).to(device)
        elif S0 == 'center':
            S0 = center(n, m)
        else:
            assert isinstance(S0, np.ndarray), "S0 must be a 2d-array, 'center', or None"
            assert S0.shape == (n + m, n + m), 'S0 must be (n+m)×(n+m)'
            assert is_in_bimapping_polytope(S0), 'S0 must be in the bi-mapping polytope'

        # Find new (approximate) solution.
    
        S, used_iter,dot_s_time, R_time, check = solve_frank_wolfe(n=n,m=m, S0=S0, c_Y_X = c_Y_X, c__Y_X = c__Y_X,
                      c__X__Y = c__X__Y, c_Y__X = c_Y__X, c_X__Y = c_X__Y, c__Y__X = c__Y__X,
                      tol=tol, max_iter=iter_budget)
        
        time_dot += dot_s_time
        time_R += R_time
        time_check += check
        
        iter_budget -= used_iter

        R = S_to_R(S, n, m)
        dis_R = dis_oles(R, X__Y, Y__X, _Y_X) * d_max

        if dis_R < best_dis_R:
            #best_f, best_g = map(list, S_to_fg(S, n, m))
            best_dis_R = dis_R


        restart_idx += 1

        # Terminate if achieved lower bound.
        if abs(best_dis_R - lb) <= 1e-1:
            break

    res = best_dis_R/2
    if (c_find == False):
        print(f'time on full frank-wolf: {time.time()-start}')

        print(f'time on full R: {time_R}')
        print(f'time on full dots: {time_dot}')
        print(f'time on check: {time_check}')
        

    return res