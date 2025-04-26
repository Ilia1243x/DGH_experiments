import numpy as np
import time
import torch

from mappings import is_row_stoch, fg_to_R
from spaces import arrange_distances

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def dot_multiplicand(S,c_Y_X, c__Y_X, c__X__Y, c_Y__X, c_X__Y, c__Y__X):
  #S = S.to(device)
  #c_Y_X = c_Y_X.to(device)
  #c__Y_X = c__Y_X.to(device)
  #c__X__Y = c__X__Y.to(device)
  #c_Y__X = c_Y__X.to(device)
  #c_X__Y = c_X__Y.to(device)
  #c__Y__X = c__Y__X.to(device)

  #print(S.dtype)
  #print(c_Y__X.dtype)

  term1 = torch.mm(torch.mm(c__Y_X, S), c_Y_X)
  term2 = torch.mm(torch.mm(c_Y_X, S), c__Y_X)
  term3 = torch.mm(torch.mm(c__X__Y, S), c_Y__X)
  term4 = torch.mm(torch.mm(c_X__Y, S), c__Y__X)
  return (term1 + term2).T + term3 + term4
  


def solve_frank_wolfe(n,m, S0, c_Y_X, c__Y_X, c__X__Y, c_Y__X, c_X__Y, c__Y__X,
                      tol=1e-16, max_iter=np.inf):
    """
    Minimizes smoothed distortion σ over the bi-mapping polytope 𝓢.

    :param obj: smoothed distortion σ:𝓢🠒ℝ (function)
    :param grad: ∇σ:𝓢🠒𝓢 (function)
    :param find_descent_direction: R:ℝ^(n+m)×(n+m)🠒𝓢 (function)
    :param minimize_obj_wrt_gamma: γ*:𝓢×𝓢🠒ℝ (function)
    :param S0: starting point in 𝓢 (2d-array)
    :param tol: tolerance for measuring rate of descent (float)
    :param max_iter: maximum number of iterations (int or ∞)
    :param verbose: no output if ≤2, iterations if >2
    :return: solution, number of iterations performed
    """
    S = S0.clone()

    dot_s_time = 0.0
    R_time = 0.0
    check = 0.0

    for iter in range(max_iter):
        # Find the Frank-Wolfe direction.

        dot_s_timei = time.time()

        dot_S = dot_multiplicand(S,c_Y_X, c__Y_X, c__X__Y, c_Y__X, c_X__Y, c__Y__X) #самая дорогая операция!

        grad_at_S = 2 * dot_S

        dot_s_time += time.time() - dot_s_timei
        
        R_timei = time.time()

        f = torch.argmin(grad_at_S[:n, :m], dim=1)

        g = torch.argmin(grad_at_S[n:, m:], dim=1)

        R = fg_to_R(f, g)

        D = R - S

        R_time += time.time() - R_timei

        time_check = time.time()

        if (iter > 0 and iter % 5 == 0):
            if torch.sum(-grad_at_S * D) < tol:
                break

        check += time.time() - time_check

        #gamma = torch.rand(1).item()
        gamma = 4 / (4 + iter)
        #gamma = min(critical_gammas, key=lambda x: np.sum((S + x*D) * dot_multiplicand(S + x*D,c,X__Y, Y__X,c_Y_X, c__Y_X)))

        # Move S towards R by γ, i.e. to (1-γ)S + γR.
        S += gamma * D

    return S, iter + 1, dot_s_time, R_time, check